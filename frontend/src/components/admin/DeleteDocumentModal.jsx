export default function DeleteDocumentModal({ isOpen, documentName, loading = false, onCancel, onDelete }) {
  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 px-4 py-6">
      <div className="w-full max-w-[440px] rounded-[16px] border border-[#eadfce] bg-white p-5 shadow-[0_30px_70px_rgba(15,23,42,0.24)]">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 flex-none items-center justify-center rounded-[12px] bg-[#fbf2df] text-[#9d7438]">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
              <path d="M6.5 7h11M10 7V5.5A1.5 1.5 0 0 1 11.5 4h1A1.5 1.5 0 0 1 14 5.5V7m-7 0 .8 11.2A1.5 1.5 0 0 0 9.3 19h5.4a1.5 1.5 0 0 0 1.5-1.8L17 7M10 11.2v4.2M14 11.2v4.2" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>

          <div className="min-w-0 flex-1">
            <h3 className="font-serif text-[20px] font-bold" style={{ color: '#111827' }}>
              Delete Document?
            </h3>
            <p className="mt-2 text-[13px] leading-6" style={{ color: '#475569' }}>
              Are you sure you want to delete this PDF? This action cannot be undone.
            </p>
            {documentName ? (
              <p className="mt-3 truncate rounded-[10px] bg-[#faf7f1] px-3 py-2 text-[12px] font-medium" style={{ color: '#334155' }} title={documentName}>
                {documentName}
              </p>
            ) : null}
          </div>
        </div>

        <div className="mt-5 flex flex-col-reverse gap-3 sm:flex-row sm:justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="inline-flex h-10 items-center justify-center rounded-[10px] border border-[#d8c5aa] bg-white px-4 text-[13px] font-medium transition hover:bg-[#fcfaf6] disabled:cursor-not-allowed disabled:opacity-60"
            style={{ color: '#334155' }}
            disabled={loading}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onDelete}
            disabled={loading}
            className="inline-flex h-10 items-center justify-center rounded-[10px] border border-[#9a5f4a] bg-[#9a5f4a] px-4 text-[13px] font-semibold text-white transition hover:bg-[#874f3d] disabled:cursor-not-allowed disabled:opacity-60"
          >
            {loading ? 'Deleting...' : 'Delete'}
          </button>
        </div>
      </div>
    </div>
  );
}
