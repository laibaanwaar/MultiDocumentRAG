export default function AuthMessage({ variant = 'info', children }) {
  if (!children) {
    return null;
  }

  const variantClassName =
    variant === 'success'
      ? 'bg-emerald-400/10 text-emerald-200'
      : variant === 'error'
        ? 'bg-rose-400/10 text-rose-200'
        : 'bg-sky-400/10 text-sky-200';

  return (
    <div className={`rounded-2xl px-3 py-2 text-[10px] ${variantClassName}`}>
      {children}
    </div>
  );
}
