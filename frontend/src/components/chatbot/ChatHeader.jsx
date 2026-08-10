import { useEffect, useRef, useState } from 'react';

function UserIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
      <circle cx="12" cy="8" r="3.2" stroke="currentColor" strokeWidth="1.7" />
      <path d="M5.5 19a6.5 6.5 0 0 1 13 0" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

export default function ChatHeader({ title, userName, onLogout }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    function handleOutsideClick(event) {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    }

    document.addEventListener('mousedown', handleOutsideClick);
    return () => {
      document.removeEventListener('mousedown', handleOutsideClick);
    };
  }, []);

  return (
    <header className="flex h-[74px] items-center justify-between border-b border-white/[0.05] px-5">
      <div className="text-[18px] font-semibold text-slate-100">{title}</div>

      <div className="flex items-center gap-3" ref={menuRef}>
        <div className="relative">
          <button
            type="button"
            onClick={() => setMenuOpen((current) => !current)}
            aria-label="Profile menu"
            className="flex h-10 w-10 items-center justify-center rounded-full bg-white/[0.03] text-slate-200 shadow-[0_10px_18px_rgba(0,0,0,0.18)]"
          >
            <UserIcon />
          </button>

          {menuOpen ? (
            <div className="absolute right-0 top-12 min-w-[150px] rounded-xl bg-[#0d1826] p-2 shadow-[0_18px_34px_rgba(0,0,0,0.35)]">
              <div className="px-2 py-2 text-[11px] text-slate-400">{userName}</div>
              <button
                type="button"
                onClick={onLogout}
                className="w-full rounded-lg px-2 py-2 text-left text-[12px] text-slate-200 transition hover:bg-white/[0.04]"
              >
                Logout
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
