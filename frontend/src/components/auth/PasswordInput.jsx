import { useState } from 'react';

function LockIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-slate-500"
      aria-hidden="true"
    >
      <rect x="5" y="11" width="14" height="9" rx="2" />
      <path d="M8 11V8a4 4 0 1 1 8 0v3" />
    </svg>
  );
}

function EyeIcon({ hidden }) {
  return hidden ? (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-slate-500"
      aria-hidden="true"
    >
      <path d="M3 3 21 21" />
      <path d="M10.58 10.58A2 2 0 0 0 13.41 13.4" />
      <path d="M9.88 5.09A9.77 9.77 0 0 1 12 4.87c5.21 0 9.27 4.13 10 7.13a11.76 11.76 0 0 1-4.08 5.55" />
      <path d="M6.61 6.61A11.74 11.74 0 0 0 2 12c.51 2.05 2.73 4.97 6.16 6.36" />
    </svg>
  ) : (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      className="text-slate-500"
      aria-hidden="true"
    >
      <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

export default function PasswordInput({
  label,
  name,
  value,
  placeholder,
  onChange,
  error,
  helper,
  autoComplete = 'current-password'
}) {
  const [showPassword, setShowPassword] = useState(false);

  return (
    <label className="block">
      <span className="mb-1 block text-[9px] font-medium text-slate-300">{label}</span>
      <div className="flex items-center gap-2 rounded-xl bg-[#111d2b]/88 px-2.5 py-2 transition focus-within:bg-[#142131]/96">
        <LockIcon />
        <input
          name={name}
          type={showPassword ? 'text' : 'password'}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          autoComplete={autoComplete}
          className="w-full border-0 bg-transparent text-[11px] text-slate-100 outline-none placeholder:text-slate-500"
        />
        <button
          type="button"
          onClick={() => setShowPassword((current) => !current)}
          className="rounded-md p-1 transition hover:bg-white/5"
          aria-label={showPassword ? 'Hide password' : 'Show password'}
        >
          <EyeIcon hidden={showPassword} />
        </button>
      </div>
      {error ? <p className="mt-1 text-[8px] text-rose-300">{error}</p> : null}
      {helper ? <p className="mt-1 text-[8px] text-slate-500">{helper}</p> : null}
    </label>
  );
}
