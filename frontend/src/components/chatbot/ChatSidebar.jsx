function ScalesIcon() {
  return (
    <svg viewBox="0 0 64 64" fill="none" aria-hidden="true" className="h-6 w-6">
      <path d="M32 10v26" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M18 18h28" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M16 22 8 35h16L16 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <path d="M48 22 40 35h16L48 22Z" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" />
      <path d="M24 46h16" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      <path d="M20 52h24" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4">
      <path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
    </svg>
  );
}

function ChatIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4">
      <path d="M21 12a8 8 0 0 1-8 8 8.7 8.7 0 0 1-3.6-.78L4 20l.9-4.48A8 8 0 1 1 21 12Z" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4">
      <path d="M3 12a9 9 0 1 0 3-6.7" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M3 4v4h4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M12 7v5l3 2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function HistoryItem({ item, isSelected, onSelect, formatTime }) {
  return (
    <button
      type="button"
      onClick={() => onSelect(item)}
      className={`w-full rounded-[10px] px-3 py-2 text-left transition ${
        isSelected ? 'bg-[#111b31] text-amber-200' : 'text-slate-300 hover:bg-[#101929]'
      }`}
    >
      <div className="truncate text-[11px] font-medium">{item.question}</div>
      <div className="mt-1 text-[9px] text-slate-500">{formatTime(item.created_at)}</div>
    </button>
  );
}

export default function ChatSidebar({
  historyItems,
  isLoadingHistory,
  selectedHistoryId,
  onSelectHistory,
  onRefreshHistory,
  onNewChat,
  formatTime
}) {
  return (
    <aside className="flex h-screen w-full max-w-[218px] flex-col bg-[linear-gradient(180deg,#0d1729_0%,#0a1220_100%)] px-4 py-4">
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-100/[0.04] text-amber-200 shadow-[0_8px_18px_rgba(0,0,0,0.18)]">
          <ScalesIcon />
        </div>
        <div>
          <div className="text-[18px] font-semibold text-slate-100">Multi Doc Chatbot</div>
          <div className="text-[11px] text-slate-400">Legal AI Assistant for Pakistan</div>
        </div>
      </div>

      <button
        type="button"
        onClick={onNewChat}
        className="mt-6 flex items-center justify-center gap-2 rounded-[10px] bg-gradient-to-b from-[#c6a35f] to-[#9f7a43] px-4 py-3 text-[14px] font-semibold text-[#f8f2e2] shadow-[0_12px_24px_rgba(173,132,63,0.22)]"
      >
        <PlusIcon />
        <span>New Chat</span>
      </button>

      <div className="mt-4 space-y-2">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center gap-3 rounded-[10px] bg-[#101a2d] px-4 py-3 text-[14px] text-amber-200 shadow-[0_10px_22px_rgba(0,0,0,0.16)]"
        >
          <ChatIcon />
          <span>New Chat</span>
        </button>

        <button
          type="button"
          onClick={onRefreshHistory}
          className="flex w-full items-center gap-3 rounded-[10px] px-2 py-2 text-[14px] text-slate-300 transition hover:bg-[#101929]"
        >
          <span className="pl-2">
            <HistoryIcon />
          </span>
          <span>Chat History</span>
        </button>
      </div>

      <div className="mt-3 flex-1 overflow-y-auto pr-1">
        {isLoadingHistory ? (
          <p className="px-2 text-[11px] text-slate-500">Loading history...</p>
        ) : historyItems.length === 0 ? (
          <p className="px-2 text-[11px] text-slate-500">Start a new conversation</p>
        ) : (
          <div className="space-y-1.5">
            {historyItems.map((item) => (
              <HistoryItem
                key={item.id}
                item={item}
                isSelected={selectedHistoryId === item.id}
                onSelect={onSelectHistory}
                formatTime={formatTime}
              />
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
