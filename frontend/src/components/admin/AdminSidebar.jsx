import { NavLink } from 'react-router-dom';

function SidebarIcon({ type }) {
  const common = 'h-5 w-5';

  if (type === 'dashboard') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <path d="M4 12.5 12 5l8 7.5V20H4v-7.5Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="M9.5 20v-5h5v5" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
      </svg>
    );
  }

  if (type === 'users') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <circle cx="9" cy="8" r="3" stroke="currentColor" strokeWidth="1.8" />
        <path d="M3.5 18a5.5 5.5 0 0 1 11 0" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M17 10.5a2.5 2.5 0 1 0 0-5M19.5 18a4.5 4.5 0 0 0-3.4-4.4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }

  if (type === 'subscriptions') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <rect x="4" y="5" width="16" height="14" rx="2" stroke="currentColor" strokeWidth="1.8" />
        <path d="M4 10h16M8 14h3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }

  if (type === 'plans') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <path d="M12 3.5c2.3 1.8 4.7 2.8 7 3v4.4c0 4.2-2.8 7.8-7 9.6-4.2-1.8-7-5.4-7-9.6V6.5c2.3-.2 4.7-1.2 7-3Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="m9.5 12.5 1.7 1.7 3.3-3.7" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    );
  }

  if (type === 'documents') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <path d="M7.5 3.75h6.2L18.5 8.4V20.25H7.5V3.75Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="M13.2 3.75V8.4h5.3" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
        <path d="M10 13h4M10 16h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
        <path d="M7 20.25H5.5A1.5 1.5 0 0 1 4 18.75V9.5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    );
  }

  if (type === 'settings') {
    return (
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
        <circle cx="12" cy="12" r="3" stroke="currentColor" strokeWidth="1.8" />
        <path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1a8.5 8.5 0 0 0-1.7-1L14.5 3h-5L9.2 6.1a8.5 8.5 0 0 0-1.7 1l-2.4-1-2 3.4 2 1.5a7 7 0 0 0 0 2l-2 1.5 2 3.4 2.4-1a8.5 8.5 0 0 0 1.7 1l.3 3.1h5l.3-3.1a8.5 8.5 0 0 0 1.7-1l2.4 1 2-3.4-2-1.5c.1-.3.1-.7.1-1Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className={common}>
      <path d="M6 6h12M6 12h12M6 18h12" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

const navItems = [
  { label: 'Dashboard', path: '/admin/dashboard', icon: 'dashboard' },
  { label: 'Users', path: '/admin/users', icon: 'users' },
  { label: 'Subscriptions', path: '/admin/subscriptions', icon: 'subscriptions' },
  { label: 'Upload PDF', path: '/admin/documents', icon: 'documents' }
];

export default function AdminSidebar({ onLogout }) {
  return (
    <aside className="flex w-full flex-col bg-[linear-gradient(180deg,#06111b_0%,#091320_100%)] px-3 py-4 text-slate-100 lg:h-screen lg:max-w-[204px] lg:sticky lg:top-0">
      <div className="border-b border-white/8 pb-4">
        <div className="font-serif text-[16px] font-semibold text-[#f5efe7]">PakLaw AI</div>
        <div className="mt-1 text-[11px] text-amber-300">Admin Panel</div>
      </div>

      <nav className="mt-5 space-y-1.5">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              [
                'flex items-center gap-3 rounded-[12px] px-3.5 py-3 text-[13px] transition',
                isActive
                  ? 'border border-[#9b723b] bg-[#162132] text-amber-200 shadow-[0_10px_20px_rgba(0,0,0,0.18)]'
                  : 'text-slate-300 hover:bg-white/[0.03] hover:text-slate-100'
              ].join(' ')
            }
          >
            <SidebarIcon type={item.icon} />
            <span>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <button
        type="button"
        onClick={onLogout}
        className="mt-auto flex items-center gap-3 rounded-[12px] border border-white/10 px-3.5 py-3 text-[13px] text-slate-200 transition hover:bg-white/[0.03]"
      >
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
          <path d="M9 4.5H6A1.5 1.5 0 0 0 4.5 6v12A1.5 1.5 0 0 0 6 19.5h3" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
          <path d="M14 8.5 18 12l-4 3.5M18 12H9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span>Logout</span>
      </button>
    </aside>
  );
}
