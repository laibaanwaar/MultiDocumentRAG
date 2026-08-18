function formatDate(value) {
  if (!value) {
    return 'N/A';
  }

  try {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    }).format(new Date(value));
  } catch {
    return 'N/A';
  }
}

function formatStatus(value) {
  if (!value) {
    return 'N/A';
  }

  return String(value)
    .replaceAll('_', ' ')
    .toLowerCase()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function statusPillClass(status) {
  const normalizedStatus = String(status || '').toLowerCase();

  if (normalizedStatus === 'ready') {
    return 'bg-emerald-50 text-emerald-700';
  }

  if (normalizedStatus === 'pending') {
    return 'bg-amber-50 text-amber-700';
  }

  if (normalizedStatus === 'failed') {
    return 'bg-rose-50 text-rose-700';
  }

  if (normalizedStatus === 'archived') {
    return 'bg-slate-100 text-slate-600';
  }

  return 'bg-slate-100 text-slate-600';
}

function EmptyState() {
  return (
    <tr className="border-t border-[#f1eadf]">
      <td colSpan={5} className="px-4 py-10 text-center">
        <p className="text-[14px] font-medium" style={{ color: '#334155' }}>
          No documents uploaded yet.
        </p>
        <p className="mt-1 text-[12px]" style={{ color: '#64748b' }}>
          Uploaded PDF documents will appear here.
        </p>
      </td>
    </tr>
  );
}

function LoadingRows() {
  return Array.from({ length: 5 }).map((_, index) => (
    <tr key={index} className="border-t border-[#f1eadf]">
      {Array.from({ length: 5 }).map((__, cellIndex) => (
        <td key={cellIndex} className="px-4 py-4">
          <div className="h-4 animate-pulse rounded bg-slate-200" />
        </td>
      ))}
    </tr>
  ));
}

export default function UploadedDocumentsTable({
  documents = [],
  loading = false,
  count = 0,
  page = 1,
  nextPage = null,
  previousPage = null,
  onNextPage,
  onPreviousPage,
  onDeleteClick
}) {
  const hasDocuments = documents.length > 0;
  const showingFrom = count === 0 ? 0 : (page - 1) * 5 + 1;
  const showingTo = count === 0 ? 0 : showingFrom + documents.length - 1;

  return (
    <section className="overflow-hidden rounded-[12px] border border-[#eadfce] bg-white shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="border-b border-[#f1eadf] px-5 py-4">
        <h2 className="font-serif text-[20px] font-bold" style={{ color: '#111827' }}>
          Uploaded Documents
        </h2>
      </div>

      <div className="overflow-x-auto">
        <table className="min-w-full table-fixed text-left">
          <thead>
            <tr className="text-[11px]" style={{ color: '#64748b' }}>
              <th className="w-[34%] px-4 py-3 font-medium">File Name</th>
              <th className="w-[22%] px-4 py-3 font-medium">Category</th>
              <th className="w-[18%] px-4 py-3 font-medium">Created Date</th>
              <th className="w-[16%] px-4 py-3 font-medium">Status</th>
              <th className="w-[10%] px-4 py-3 font-medium">Actions</th>
            </tr>
          </thead>

          <tbody className="text-[12px]" style={{ color: '#334155' }}>
            {loading ? <LoadingRows /> : null}

            {!loading && !hasDocuments ? <EmptyState /> : null}

            {!loading
              ? documents.map((document) => (
                  <tr key={document.id} className="border-t border-[#f1eadf] align-middle">
                    <td className="px-4 py-4">
                      <div className="truncate text-[12px] font-semibold" style={{ color: '#111827' }} title={document.original_filename}>
                        {document.original_filename || 'N/A'}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <div className="truncate text-[12px]" style={{ color: '#475569' }} title={document.category?.name}>
                        {document.category?.name || 'N/A'}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <div className="text-[12px]" style={{ color: '#475569' }}>
                        {formatDate(document.created_at)}
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusPillClass(document.status)}`}>
                        {formatStatus(document.status)}
                      </span>
                    </td>

                    <td className="px-4 py-4">
                      <button
                        type="button"
                        onClick={() => onDeleteClick?.(document)}
                        className="inline-flex h-9 items-center justify-center rounded-[10px] border border-[#e4c7b4] px-3 text-[12px] font-medium text-[#9a5f4a] transition hover:bg-[#fcf6f1]"
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))
              : null}
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-3 border-t border-[#f1eadf] px-5 py-4 text-[13px] text-slate-500 sm:flex-row sm:items-center sm:justify-between">
        <p>
          Showing {showingFrom} to {showingTo} of {count} results
        </p>

        <div className="flex items-center gap-2">
          <button
            type="button"
            disabled={!previousPage || loading}
            onClick={onPreviousPage}
            className="flex h-8 w-8 items-center justify-center rounded-[8px] border border-[#d8c5aa] text-slate-700 transition disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Previous page"
          >
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4">
              <path d="M14.5 6 8.5 12l6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>

          <span className="flex h-8 min-w-[34px] items-center justify-center rounded-[8px] border border-[#d8c5aa] px-3 text-[12px] font-semibold text-slate-700">
            {page}
          </span>

          <button
            type="button"
            disabled={!nextPage || loading}
            onClick={onNextPage}
            className="flex h-8 w-8 items-center justify-center rounded-[8px] border border-[#d8c5aa] text-slate-700 transition disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="Next page"
          >
            <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4">
              <path d="M9.5 6 15.5 12l-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </button>
        </div>
      </div>
    </section>
  );
}
