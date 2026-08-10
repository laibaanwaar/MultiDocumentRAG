function BotIcon() {
  return (
    <div className="flex h-9 w-9 items-center justify-center rounded-full bg-white/[0.04] text-amber-200 shadow-[0_10px_18px_rgba(0,0,0,0.16)]">
      <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4.5 w-4.5">
        <circle cx="12" cy="12" r="7.5" stroke="currentColor" strokeWidth="1.6" />
        <path d="M9.5 10h.01M14.5 10h.01" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
        <path d="M9 14c.8.7 1.8 1 3 1s2.2-.3 3-1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      </svg>
    </div>
  );
}

function SourceList({ sources }) {
  if (!Array.isArray(sources) || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 space-y-1.5">
      {sources.map((source, index) => (
        <div key={`${index}-${source?.label || source?.document_title || 'source'}`} className="text-[10px] text-slate-400">
          {source?.document_title || source?.document_name || source?.label || `Source ${index + 1}`}
        </div>
      ))}
    </div>
  );
}

function UserMessage({ text, time }) {
  return (
    <div className="ml-auto flex max-w-[64%] flex-col items-end">
      <div className="rounded-[14px] bg-[#212c56] px-4 py-3 text-[13px] text-slate-100 shadow-[0_10px_24px_rgba(0,0,0,0.18)]">
        {text}
      </div>
      <div className="mt-2 text-[10px] text-slate-500">{time}</div>
    </div>
  );
}

function AssistantMessage({ text, time, sources }) {
  return (
    <div className="flex max-w-[78%] items-start gap-3">
      <BotIcon />
      <div>
        <div className="rounded-[14px] bg-white/[0.04] px-4 py-3 text-[13px] leading-6 text-slate-100 shadow-[0_12px_24px_rgba(0,0,0,0.18)]">
          <div>{text}</div>
          <SourceList sources={sources} />
        </div>
        <div className="mt-2 text-[10px] text-slate-500">{time}</div>
      </div>
    </div>
  );
}

export default function ChatWindow({
  selectedHistory,
  pendingQuestion,
  pendingAnswer,
  pendingSources,
  isAnswering,
  formatTime
}) {
  const questionText = selectedHistory ? selectedHistory.question : pendingQuestion;
  const answerText = selectedHistory ? selectedHistory.answer : pendingAnswer;
  const sources = selectedHistory ? selectedHistory.sources : pendingSources;
  const time = selectedHistory ? selectedHistory.created_at : new Date().toISOString();

  if (!questionText && !answerText && !isAnswering) {
    return null;
  }

  return (
    <div className="mt-4 space-y-5 pb-8">
      {questionText ? <UserMessage text={questionText} time={formatTime(time)} /> : null}
      {answerText || isAnswering ? (
        <AssistantMessage
          text={isAnswering && !answerText ? 'Thinking...' : answerText}
          time={formatTime(time)}
          sources={sources}
        />
      ) : null}
    </div>
  );
}
