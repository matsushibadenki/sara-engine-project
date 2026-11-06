"""Model-free cooperative lifecycle gate; not a live model authority or writer."""
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock, get_ident


class HostError(ValueError):
    """Rejected lifecycle request; never authorizes model operations."""


class HostBusy(HostError):
    """Another mutation or pause already owns the gate."""


@dataclass(frozen=True, slots=True)
class Owner:
    label: str
    epoch_label: str


@dataclass(frozen=True, slots=True)
class View:
    owner_label: str
    epoch_label: str
    revision: int
    phase: str


@dataclass(frozen=True, slots=True)
class PauseTicket:
    view: View


class OwnerHost:
    """Contains control metadata only; no model, callable, restore or storage API.

    Capabilities use object identity within one cooperative process. Python
    private attributes and hand labels do not authenticate an external owner.
    """

    def __init__(self, owner_label, epoch_label):
        for label in (owner_label, epoch_label):
            if type(label) is not str or not 0 < len(label) <= 128 or label.strip() != label:
                raise HostError('Invalid owner/epoch label')
        self._owner = Owner(owner_label, epoch_label)
        self._gate = Lock()
        self._revision = 0
        self._phase = 'stable'
        self._ticket = None
        self._pause_thread = None

    @classmethod
    def create(cls, owner_label, epoch_label):
        host = cls(owner_label, epoch_label)
        return host, host._owner

    def _check_owner(self, owner):
        if owner is not self._owner:
            raise HostError('Foreign owner capability')

    def _view(self):
        return View(self._owner.label, self._owner.epoch_label, self._revision, self._phase)

    def _stable(self, revision):
        if type(revision) is not int or revision < 0:
            raise HostError('Expected revision must be an exact nonnegative integer')
        if self._phase != 'stable' or self._revision != revision:
            raise HostError('Closed, poisoned or stale host')

    @contextmanager
    def _exclusive(self, owner):
        self._check_owner(owner)
        if not self._gate.acquire(blocking=False):
            raise HostBusy('Exclusive host gate is busy')
        try:
            yield
        finally:
            self._gate.release()

    def inspect(self, owner):
        """Read terminal/stable metadata; does not recover or authorize a retry."""
        with self._exclusive(owner):
            return self._view()

    @contextmanager
    def mutation(self, owner, expected_revision):
        """Mark a metadata-only mutation interval; execute no callback or payload."""
        with self._exclusive(owner):
            self._stable(expected_revision)
            self._revision += 1
            self._phase = 'mutating'
            try:
                yield self._view()
            except BaseException:
                self._phase = 'poisoned'
                raise
            else:
                self._revision += 1
                self._phase = 'stable'

    @contextmanager
    def paused(self, owner, expected_revision):
        """Hold the same exclusive gate through all repeated ticket inspections."""
        with self._exclusive(owner):
            self._stable(expected_revision)
            ticket = PauseTicket(self._view())
            self._ticket, self._pause_thread = ticket, get_ident()
            try:
                yield ticket
            finally:
                self._ticket, self._pause_thread = None, None

    def revalidate(self, owner, ticket):
        """Check an active same-thread ticket without reacquiring or changing state."""
        self._check_owner(owner)
        if (ticket is not self._ticket or self._pause_thread != get_ident()
                or type(ticket) is not PauseTicket):
            raise HostError('Expired, forged or foreign-thread pause ticket')
        if ticket.view != self._view() or self._phase != 'stable':
            raise HostError('Paused metadata changed')
        return self._view()

    def close(self, owner, expected_revision):
        with self._exclusive(owner):
            self._stable(expected_revision)
            self._phase = 'closed'
            return self._view()
