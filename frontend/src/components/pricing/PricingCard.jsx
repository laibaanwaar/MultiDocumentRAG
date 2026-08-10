function CheckIcon() {
  return (
    <span className="mt-[2px] inline-flex h-4.5 w-4.5 flex-none items-center justify-center rounded-full bg-[#f1c36c] text-[#0d1826]">
      <svg viewBox="0 0 20 20" fill="none" aria-hidden="true" className="h-2.5 w-2.5">
        <path d="m5.2 10.1 3.1 3.2 6.5-6.6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

export default function PricingCard({
  name,
  subtitle,
  price,
  buttonLabel,
  features,
  highlighted = false
}) {
  return (
    <article
      className={[
        'relative flex h-full min-h-[340px] flex-col rounded-[12px] bg-[linear-gradient(180deg,#0b1726_0%,#09131f_100%)] px-5 py-5 shadow-[0_16px_34px_rgba(0,0,0,0.24)]',
        highlighted ? 'border border-[#c99c47]' : 'border border-white/10'
      ].join(' ')}
    >
      {highlighted ? (
        <div className="absolute left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 rounded-[9px] bg-gradient-to-r from-[#f3ce86] to-[#ddb05d] px-4 py-1 text-[12px] font-semibold text-[#3a2a12]">
          Most Popular
        </div>
      ) : null}

      <div>
        <h3 className="font-serif text-[23px] leading-none text-[#f4efe6]">{name}</h3>
        <p className="mt-2 text-[13px] text-slate-400">{subtitle}</p>
      </div>

      <div className="mt-5 flex items-end gap-2">
        <span className="font-serif text-[24px] font-semibold leading-none text-[#e0b160]">{price}</span>
        <span className="text-[13px] text-slate-400">/ month</span>
      </div>

      <div className="mt-5 border-t border-white/8 pt-5">
        <ul className="space-y-3">
          {features.map((feature) => (
            <li key={feature} className="flex items-start gap-2.5 text-[13px] leading-5 text-slate-200">
              <CheckIcon />
              <span>{feature}</span>
            </li>
          ))}
        </ul>
      </div>

      <button
        type="button"
        className={[
          'mt-auto rounded-[10px] px-4 py-3 text-[14px] font-semibold transition',
          highlighted
            ? 'bg-gradient-to-r from-[#f5d18f] to-[#dfb25f] text-[#121212] shadow-[0_14px_30px_rgba(223,178,95,0.18)] hover:brightness-105'
            : 'border border-[#8b6834] text-[#eac679] hover:bg-[#161f2b]'
        ].join(' ')}
      >
        {buttonLabel}
      </button>
    </article>
  );
}
