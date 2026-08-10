function AskIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="h-11 w-11">
      <path d="M14 31.5V37l6.4-4.3h9.8c5.4 0 9.8-4.2 9.8-9.4S35.6 14 30.2 14H17.8C12.4 14 8 18.2 8 23.3c0 3.3 1.8 6.2 4.6 7.9Z" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M18.2 23.3h.1M24 23.3h.1M29.8 23.3h.1" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
    </svg>
  );
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="h-11 w-11">
      <path d="M10 10.5h16l8 8V38a2 2 0 0 1-2 2H10a2 2 0 0 1-2-2v-25.5a2 2 0 0 1 2-2Z" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M26 10.5V19h8" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <circle cx="27.5" cy="29.5" r="5.5" stroke="currentColor" strokeWidth="2.2" />
      <path d="m31.6 33.6 5.4 5.4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
      <path d="M14.5 21.5h7M14.5 27.5h4" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
    </svg>
  );
}

function AnswerIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="h-11 w-11">
      <path d="M24 8.5c4.8 4.1 10.8 6.2 18 6.4v8.8c0 10.1-7.3 18.5-18 20.8C13.3 42.2 6 33.8 6 23.7v-8.8c7.2-.2 13.2-2.3 18-6.4Z" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="m18 24.5 4.3 4.3 7.7-9" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="h-11 w-11">
      <path d="M9 13v10h10" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M13 32.5A16 16 0 1 0 9 23" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M24 16.5v8l5.2 3.3" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

const icons = {
  ask: AskIcon,
  search: SearchIcon,
  answer: AnswerIcon,
  history: HistoryIcon
};

export default function ProcessStepCard({ number, title, description, icon, showArrow = false }) {
  const Icon = icons[icon];

  return (
    <div className="relative h-full">
      <article className="relative flex h-[190px] flex-col rounded-[10px] border border-[#9a7137] bg-[linear-gradient(180deg,#0b1726_0%,#09121f_100%)] px-5 py-5 shadow-[0_14px_32px_rgba(0,0,0,0.26)]">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-b from-[#f4cf87] to-[#d7a94f] text-[18px] font-semibold text-[#24170a]">
          {number}
        </div>

        <div className="mt-3 flex flex-1 flex-col items-center text-center">
          <div className="text-[#e0b160]">
            <Icon />
          </div>
          <div className="mt-3 h-[2px] w-6 rounded-full bg-[#c39343]" />
          <h3 className="mt-3 font-serif text-[17px] leading-6 text-[#f3eee4]">{title}</h3>
          <p className="mt-1.5 max-w-[220px] text-[13px] leading-5 text-slate-400">{description}</p>
        </div>
      </article>

      {showArrow ? (
        <div className="pointer-events-none absolute -right-5 top-1/2 hidden -translate-y-1/2 xl:flex items-center justify-center text-[#d2a14e]">
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-6 w-6">
            <path d="M6 4.5 14 12l-8 7.5" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </div>
      ) : null}
    </div>
  );
}
