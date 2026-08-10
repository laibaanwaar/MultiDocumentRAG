export default function AuthButton({
  children,
  type = 'button',
  variant = 'primary',
  disabled = false,
  onClick
}) {
  const baseClassName =
    'flex w-full items-center justify-center gap-2 rounded-xl px-4 py-2 text-[11px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-70';

  const variantClassName =
    variant === 'secondary'
      ? 'bg-white/3 text-slate-200 hover:bg-white/6'
      : 'bg-gradient-to-r from-[#f3ce86] to-[#e4b765] text-slate-950 shadow-[0_12px_22px_rgba(228,183,101,0.18)] hover:brightness-105';

  return (
    <button
      type={type}
      disabled={disabled}
      onClick={onClick}
      className={`${baseClassName} ${variantClassName}`}
    >
      {children}
    </button>
  );
}
