import { useNavigate } from 'react-router-dom';
import SiteHeader from '../components/layout/SiteHeader';
import ProcessStepCard from '../components/how-it-works/ProcessStepCard';

const steps = [
  {
    number: 1,
    icon: 'ask',
    title: 'Ask Your Question',
    description: 'Type your legal question in plain language.'
  },
  {
    number: 2,
    icon: 'search',
    title: 'AI Searches Documents',
    description: 'The system scans relevant Pakistani laws and provisions.'
  },
  {
    number: 3,
    icon: 'answer',
    title: 'Get Accurate Answer',
    description: 'Receive a clear answer with legal references and sources.'
  },
  {
    number: 4,
    icon: 'history',
    title: 'Review History',
    description: 'Revisit previous questions and continue your research anytime.'
  }
];

const processStages = [
  { label: 'Question', icon: '💬' },
  { label: 'Search', icon: '🔍' },
  { label: 'Answer', icon: '🛡️' },
  { label: 'Sources', icon: '📄' }
];

function CenterMark() {
  return (
    <div className="flex items-center justify-center gap-5">
      <div className="h-px w-16 bg-[#9d7438]" />
      <div className="text-[#e0b160]">
        <svg viewBox="0 0 48 48" fill="none" aria-hidden="true" className="h-8 w-8">
          <path d="M24 8v20" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
          <path d="M13 14h22" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
          <path d="M12 17.5 6 28h12l-6-10.5Z" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
          <path d="M36 17.5 30 28h12l-6-10.5Z" stroke="currentColor" strokeWidth="2.2" strokeLinejoin="round" />
          <path d="M18 36h12" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
        </svg>
      </div>
      <div className="h-px w-16 bg-[#9d7438]" />
    </div>
  );
}

function ProcessBar() {
  return (
    <div className="mx-auto mt-4 max-w-[720px] rounded-[12px] border border-white/8 bg-[linear-gradient(180deg,#0b1726_0%,#0a1420_100%)] px-5 py-4 shadow-[0_12px_26px_rgba(0,0,0,0.22)]">
      <div className="grid gap-4 md:grid-cols-4">
        {processStages.map((stage, index) => (
          <div key={stage.label} className="flex items-center justify-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-full border border-[#9a7137] bg-[#0d1928] text-[16px]">
              {stage.icon}
            </div>
            <div className="text-[14px] text-[#f3eee4]">{stage.label}</div>
            {index < processStages.length - 1 ? (
              <div className="hidden flex-1 items-center md:flex">
                <div className="h-px w-full bg-[#b5863e]" />
                <div className="ml-1 h-1.5 w-1.5 rounded-full bg-[#d1a04d]" />
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function HowItWorksPage() {
  const navigate = useNavigate();

  return (
    <>
      <SiteHeader activeItem="How It Works" />

      <main className="bg-[radial-gradient(circle_at_top,rgba(212,160,72,0.08),transparent_18%),linear-gradient(180deg,#06111d_0%,#081321_100%)] px-4 py-8 sm:px-6 lg:px-10 lg:py-9">
        <section className="mx-auto max-w-[1200px]">
          <div className="text-center">
            <CenterMark />
            <h1 className="mt-4 font-serif text-[clamp(2rem,3.8vw,3.35rem)] font-semibold leading-[1.05] text-[#f5efe7]">
              How PakLaw AI <span className="text-[#d7a655]">Works</span>
            </h1>
            <p className="mx-auto mt-3 max-w-[720px] text-[15px] leading-6 text-slate-400">
              Ask, analyze, and get trusted answers from Pakistani legal documents in just a few steps.
            </p>
          </div>

          <div className="mt-8 grid gap-5 md:grid-cols-2 xl:grid-cols-4">
            {steps.map((step, index) => (
              <ProcessStepCard
                key={step.number}
                {...step}
                showArrow={index < steps.length - 1}
              />
            ))}
          </div>

          <ProcessBar />

          <div className="mt-5 flex flex-col items-center justify-center gap-3 sm:flex-row">
            <button
              type="button"
              onClick={() => navigate('/signup')}
              className="rounded-[10px] bg-gradient-to-r from-[#f6d59d] to-[#e0b160] px-7 py-3 text-[14px] font-semibold text-slate-950 shadow-[0_14px_28px_rgba(224,177,96,0.2)] transition hover:brightness-105"
            >
              Start Asking Now →
            </button>
            <button
              type="button"
              onClick={() => navigate('/#features')}
              className="rounded-[10px] border border-[#9a7137] px-7 py-3 text-[14px] font-medium text-slate-100 transition hover:bg-white/[0.03]"
            >
              View Features
            </button>
          </div>
        </section>
      </main>
    </>
  );
}
