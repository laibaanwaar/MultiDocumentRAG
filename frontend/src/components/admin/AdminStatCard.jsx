export default function AdminStatCard({ label, value, loading = false }) {
  return (
    <article className="min-h-[118px] rounded-[12px] border border-[#eadfce] bg-white px-5 py-5 shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="flex items-center gap-4">
        <div className="flex h-12 w-12 flex-none items-center justify-center rounded-[12px] bg-[#fbf2df] text-[#9d7438]">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-6 w-6">
            <path d="M4 18a4 4 0 0 1 4-4h8a4 4 0 0 1 4 4M9 10a3 3 0 1 1 0-6 3 3 0 0 1 0 6ZM17 11a2.5 2.5 0 1 0 0-5" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>

        <div className="flex min-w-0 flex-col justify-center">
          <div
            className="block text-[13px] font-bold leading-5"
            style={{ color: '#64748b', display: 'block', visibility: 'visible', opacity: 1 }}
          >
            {label}
          </div>

          {loading ? (
            <div className="mt-3 h-8 w-20 animate-pulse rounded bg-slate-200" />
          ) : (
            <div
              className="mt-2 block font-serif text-[31px] leading-none"
              style={{ color: '#111827', display: 'block', visibility: 'visible', opacity: 1 }}
            >
              {value ?? '—'}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}
