function ScalesTile() {
  return (
    <div className="flex h-[54px] w-[54px] items-center justify-center rounded-[14px] bg-white/[0.04] text-amber-200 shadow-[0_10px_20px_rgba(0,0,0,0.16)]">
      <svg viewBox="0 0 64 64" fill="none" aria-hidden="true" className="h-7 w-7">
        <path d="M32 10v26" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M18 18h28" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M16 22 8 35h16L16 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
        <path d="M48 22 40 35h16L48 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
        <path d="M24 46h16" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
        <path d="M20 52h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
    </div>
  );
}

export default function DocumentList({ documents }) {
  const hasDocuments = Array.isArray(documents) && documents.length > 0;

  return (
    <div className="pt-7">
      <div className="flex items-start gap-3">
        <ScalesTile />
        <div className="pt-1">
          <div className="font-serif text-[21px] text-[#f3efe6]">
            Welcome to <span className="text-amber-300">Multi Doc Chatbot!</span>
          </div>
          <p className="mt-1 text-[13px] text-slate-400">Upload your documents and ask anything.</p>
        </div>
      </div>

      <div className="mt-4 max-w-[370px] rounded-[14px] bg-white/[0.025] px-4 py-4 shadow-[0_16px_30px_rgba(0,0,0,0.16)]">
        <div className="text-[13px] font-semibold text-amber-200">
          Loaded Documents{hasDocuments ? ` (${documents.length})` : ''}
        </div>

        {hasDocuments ? (
          <div className="mt-3 space-y-2">
            {documents.map((document) => (
              <div key={document.id} className="text-[12px] text-slate-300">
                {document.title || document.original_filename || document.name || 'Document'}
              </div>
            ))}
          </div>
        ) : (
          <p className="mt-3 text-[12px] text-slate-400">No documents available</p>
        )}
      </div>
    </div>
  );
}
