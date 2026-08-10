export default function AuthMessage({ variant = 'info', children }) {
  if (!children) {
    return null;
  }

  const variantClassName =
    variant === 'success'
      ? 'border border-emerald-200 bg-emerald-50 text-emerald-700'
      : variant === 'error'
        ? 'border border-rose-200 bg-rose-50 text-rose-700'
        : 'border border-sky-200 bg-sky-50 text-sky-700';

  return (
    <div className={`rounded-2xl px-3 py-2 text-[12px] ${variantClassName}`}>
      {children}
    </div>
  );
}
