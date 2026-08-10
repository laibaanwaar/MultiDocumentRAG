function AttachmentIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
      <path d="M8.5 12.5 14.9 6.1a2.8 2.8 0 1 1 4 4L10.7 18.3a5 5 0 1 1-7.1-7.1l8-8" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-5 w-5">
      <path d="M21 3 10 14" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
      <path d="m21 3-7 18-4-7-7-4 18-7Z" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function ChatInput({ value, onChange, onSubmit, disabled, error }) {
  return (
    <div className="px-4 pb-4 pt-3">
      {error ? <div className="mb-2 text-[11px] text-rose-300">{error}</div> : null}

      <form
        onSubmit={onSubmit}
        className="flex items-center gap-3 rounded-[15px] bg-[#0d1729] px-4 py-3 shadow-[0_14px_28px_rgba(0,0,0,0.18)]"
      >
        <span className="text-slate-500">
          <AttachmentIcon />
        </span>
        <input
          value={value}
          onChange={onChange}
          placeholder="Type your question here..."
          className="flex-1 border-0 bg-transparent text-[13px] text-slate-100 outline-none placeholder:text-slate-500"
        />
        <button
          type="submit"
          disabled={disabled}
          className="flex h-9 w-9 items-center justify-center rounded-[10px] bg-gradient-to-b from-[#d1ae68] to-[#a47b42] text-[#111827] shadow-[0_10px_20px_rgba(164,123,66,0.24)] disabled:cursor-not-allowed disabled:opacity-70"
        >
          <SendIcon />
        </button>
      </form>
    </div>
  );
}
