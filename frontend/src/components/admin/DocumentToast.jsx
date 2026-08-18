export default function DocumentToast({ message, variant = 'info', onClose }) {
  if (!message) {
    return null;
  }

  const variantClassName =
    variant === 'success'
      ? 'border-[#d1eadf] bg-[#eff9f3] text-[#165b39]'
      : variant === 'error'
        ? 'border-[#ebd1d1] bg-[#fdf2f2] text-[#8a2f2f]'
        : 'border-[#eadfce] bg-white text-[#334155]';

  return (
    <div className="fixed right-4 top-4 z-50 w-[calc(100vw-2rem)] max-w-[420px]">
      <div className={`rounded-[14px] border px-4 py-3 shadow-[0_18px_38px_rgba(15,23,42,0.16)] ${variantClassName}`}>
        <div className="flex items-start gap-3">
          <div className="mt-0.5 flex h-8 w-8 flex-none items-center justify-center rounded-full bg-white/80 text-[12px] font-bold">
            {variant === 'success' ? 'OK' : variant === 'error' ? '!' : 'i'}
          </div>

          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold">{variant === 'success' ? 'Success' : variant === 'error' ? 'Error' : 'Notice'}</div>
            <p className="mt-1 text-[12px] leading-5">{message}</p>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="ml-1 flex h-7 w-7 flex-none items-center justify-center rounded-full text-[14px] font-semibold transition hover:bg-black/5"
            aria-label="Dismiss notification"
          >
            ×
          </button>
        </div>
      </div>
    </div>
  );
}
