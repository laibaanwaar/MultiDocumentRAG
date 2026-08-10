function FieldGlyph({ kind }) {
  const common = {
    width: 18,
    height: 18,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    strokeWidth: '1.8',
    strokeLinecap: 'round',
    strokeLinejoin: 'round',
    className: 'text-slate-500'
  };

  if (kind === 'mail') {
    return (
      <svg {...common}>
        <path d="M4 7h16v10H4z" />
        <path d="m4 8 8 6 8-6" />
      </svg>
    );
  }

  if (kind === 'username') {
    return (
      <svg {...common}>
        <circle cx="12" cy="12" r="8" />
        <path d="M16 8a4 4 0 1 0 0 8h2" />
      </svg>
    );
  }

  return (
    <svg {...common}>
      <circle cx="12" cy="8" r="3.2" />
      <path d="M5.5 19a6.5 6.5 0 0 1 13 0" />
    </svg>
  );
}

export default function AuthField({
  label,
  name,
  type = 'text',
  value,
  placeholder,
  onChange,
  icon = 'user',
  error,
  helper,
  autoComplete = 'off',
  readOnly = false
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[9px] font-medium text-slate-300">{label}</span>
      <div className="flex items-center gap-2 rounded-xl bg-[#111d2b]/88 px-2.5 py-2 transition focus-within:bg-[#142131]/96">
        <FieldGlyph kind={icon} />
        <input
          name={name}
          type={type}
          value={value}
          onChange={onChange}
          placeholder={placeholder}
          autoComplete={autoComplete}
          readOnly={readOnly}
          className="w-full border-0 bg-transparent text-[11px] text-slate-100 outline-none placeholder:text-slate-500 read-only:text-slate-300"
        />
      </div>
      {error ? <p className="mt-1 text-[8px] text-rose-300">{error}</p> : null}
      {helper ? <p className="mt-1 text-[8px] text-slate-500">{helper}</p> : null}
    </label>
  );
}
