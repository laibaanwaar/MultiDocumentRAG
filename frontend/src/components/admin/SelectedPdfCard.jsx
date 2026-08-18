function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return '0 B';
  }

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export default function SelectedPdfCard({ file, onSubmit, loading = false, disabled = false }) {
  if (!file) {
    return null;
  }

  const isDisabled = disabled || loading;

  return (
    <section className="rounded-[12px] border border-[#eadfce] bg-white px-4 py-4 shadow-[0_8px_18px_rgba(15,23,42,0.04)] sm:px-5">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-start gap-3">
          <div className="flex h-11 w-11 flex-none items-center justify-center rounded-[12px] bg-[#fbf2df] text-[#9d7438]">
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
              <path d="M7.5 3.75h6.2L18.5 8.4V20.25H7.5V3.75Z" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
              <path d="M13.2 3.75V8.4h5.3" stroke="currentColor" strokeWidth="1.8" strokeLinejoin="round" />
              <path d="M10 13h4M10 16h4" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
            </svg>
          </div>

          <div className="min-w-0">
            <h3 className="font-serif text-[17px] font-bold" style={{ color: '#111827' }}>
              Selected PDF File
            </h3>
            <p className="mt-1 truncate text-[13px] font-medium" style={{ color: '#334155' }} title={file.name}>
              {file.name}
            </p>
            <p className="mt-1 text-[12px]" style={{ color: '#64748b' }}>
              {formatFileSize(file.size)}
            </p>
          </div>
        </div>

        <button
          type="button"
          onClick={onSubmit}
          disabled={isDisabled}
          className="inline-flex h-10 items-center justify-center rounded-[10px] border border-[#9d7438] bg-[#9d7438] px-4 text-[13px] font-semibold text-white transition hover:bg-[#8d6731] disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading ? 'Uploading...' : 'Submit Document'}
        </button>
      </div>
    </section>
  );
}
