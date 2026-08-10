import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import heroBg from '../../assets/law-background.png';
import SiteHeader from '../components/layout/SiteHeader';

const stats = [
  { value: '25+', label: 'Legal Documents', icon: 'DOC' },
  { value: 'AI-Powered', label: 'Smart Answers', icon: 'AI' },
  { value: '100%', label: 'Secure & Private', icon: 'SAFE' }
];

const featureCards = [
  {
    title: 'Multi-Document AI Search',
    description: 'Search across multiple legal documents simultaneously and get precise, relevant answers.'
  },
  {
    title: 'Laws of Pakistan',
    description: 'Built specifically for Pakistani legal system with updated acts, codes and regulations.'
  },
  {
    title: 'AI-Powered Accuracy',
    description: 'Advanced AI understands context and provides accurate, reliable legal information.'
  },
  {
    title: 'Secure & Confidential',
    description: 'Your queries and documents are encrypted and kept completely secure.'
  },
  {
    title: 'Chat History',
    description: 'Save, organize and revisit your conversations and legal research anytime.'
  }
];

const suggestions = ['What is bail under CrPC?', 'Explain Section 302 of PPC', 'What is a contract under law?'];

function StatItem({ icon, value, label }) {
  return (
    <div className="flex items-start gap-2.5">
      <div className="mt-1 text-[11px] font-semibold uppercase tracking-[0.18em] text-amber-300">
        {icon}
      </div>
      <div>
        <div className="text-[13px] font-semibold text-amber-200 md:text-[14px]">{value}</div>
        <div className="mt-1 text-[11px] text-slate-400">{label}</div>
      </div>
    </div>
  );
}

function FeatureCard({ title, description }) {
  return (
    <article className="flex min-h-[188px] flex-col items-center rounded-[16px] border border-slate-200 bg-white px-4 py-5 text-center shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="mx-auto mb-4 flex h-10 w-10 items-center justify-center rounded-full bg-amber-50 text-[11px] font-semibold text-amber-700 shadow-inner">
        i
      </div>
      <div className="mx-auto max-w-[186px]">
        <h3 className="text-[12px] font-semibold leading-5 text-slate-900">{title}</h3>
        <p className="mt-2 text-[11px] leading-5 text-slate-600">{description}</p>
      </div>
    </article>
  );
}

function ChatBubble({ children, user = false }) {
  return (
    <div
      className={[
        'max-w-[86%] rounded-xl px-3.5 py-2.5 text-[13px] leading-5 shadow-lg',
        user
          ? 'ml-auto bg-gradient-to-r from-[#72532f] to-[#8d6838] text-stone-100'
          : 'bg-white/10 text-slate-100 backdrop-blur-sm'
      ].join(' ')}
    >
      {children}
    </div>
  );
}

export default function Home() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash === '#features') {
      const target = document.getElementById('features');
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    }
  }, [location.hash]);

  return (
    <>
      <SiteHeader activeItem="Home" />

      <main>
        <section
          className="relative isolate overflow-hidden border-b border-black/20"
          style={{
            backgroundImage: `linear-gradient(90deg, rgba(5,10,18,0.98) 0%, rgba(5,10,18,0.94) 42%, rgba(5,10,18,0.72) 63%, rgba(5,10,18,0.35) 100%), url(${heroBg})`,
            backgroundPosition: 'right center',
            backgroundRepeat: 'no-repeat',
            backgroundSize: 'cover'
          }}
        >
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(249,184,82,0.06),transparent_24%),radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.03),transparent_24%)]" />

          <div className="mx-auto grid min-h-[660px] max-w-[1440px] items-start gap-6 px-4 py-4 sm:px-6 lg:grid-cols-[0.9fr_0.78fr] lg:items-start lg:px-10 lg:py-6">
            <div className="relative z-10 max-w-[520px] pt-2 lg:pt-4">
              <h1 className="hero-title mt-5 max-w-[500px] font-serif text-[clamp(1.95rem,3.1vw,2.9rem)] font-semibold leading-[0.98] tracking-tight text-[#f5efe7]">
                <span className="hero-line block">Your Intelligent Legal</span>
                <span className="hero-line hero-line-delay-1 block">Research</span>
                <span className="hero-line hero-line-delay-2 block text-[#d9ad63]">Companion</span>
              </h1>

              <p className="mt-4 max-w-[480px] text-[13px] leading-6 text-slate-300">
                Ask questions, get accurate answers, and find relevant information from multiple
                legal documents based on the laws of Pakistan.
              </p>

              <div className="mt-8 grid gap-4 sm:grid-cols-3">
                {stats.map((stat) => (
                  <StatItem key={stat.label} {...stat} />
                ))}
              </div>
            </div>

            <div className="relative z-10 flex items-start justify-center lg:justify-end lg:pt-4 lg:mr-14 xl:mr-24">
              <div className="hero-chat-shell relative w-full max-w-[380px]">
                <div className="relative rounded-[22px] border border-[#d6b06a]/20 bg-[#101c2d]/82 p-2.5 shadow-[0_20px_60px_rgba(0,0,0,0.45)] backdrop-blur-xl">
                  <div className="rounded-[18px] border border-white/5 bg-white/5 p-3.5">
                    <div className="flex items-center gap-3">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl border border-amber-200/20 bg-amber-100/10 text-[13px] font-semibold text-amber-200">
                        LA
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="text-[14px] font-semibold text-slate-100">PakLaw AI</span>
                          <span className="text-xs font-medium text-emerald-400">Online</span>
                        </div>
                        <div className="text-[11px] text-slate-400">Ask anything about Pakistani Laws</div>
                      </div>
                    </div>

                    <div className="mt-3.5 space-y-3">
                      <ChatBubble user>
                        What is the punishment for theft under Section 378 of PPC?
                        <div className="mt-2 text-right text-[9px] text-stone-200/60">10:30 AM</div>
                      </ChatBubble>

                      <ChatBubble>
                        Under Section 378 of the Pakistan Penal Code, the punishment for theft is
                        imprisonment for a term which may extend to three years, or with fine, or with
                        both.
                        <div className="mt-2.5 text-[9px] text-slate-300">10:31 AM</div>
                        <div className="mt-2 inline-flex items-center gap-1 text-[10px] text-amber-200/80">
                          <span>Source:</span>
                          <span>Pakistan Penal Code, 1860</span>
                        </div>
                      </ChatBubble>

                      <div>
                        <div className="mb-2.5 text-[10px] font-medium uppercase tracking-[0.22em] text-slate-400">
                          Try asking
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {suggestions.map((item) => (
                            <span
                              key={item}
                              className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-[10px] text-slate-300"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>

                      <div className="flex items-center gap-3 rounded-xl border border-white/10 bg-white/5 px-3.5 py-2">
                        <span className="text-[11px] text-slate-400">Ask a legal question...</span>
                        <button className="ml-auto rounded-lg bg-gradient-to-r from-[#f6d59d] to-[#e1b86d] px-3 py-2 text-[11px] font-semibold text-slate-950">
                          Send
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="features" className="border-t border-slate-200 bg-[#f8f3ea] px-4 py-5 sm:px-6 lg:px-10 lg:py-6">
          <div className="mx-auto grid max-w-[1500px] gap-3 md:grid-cols-2 xl:grid-cols-5">
            {featureCards.map((card) => (
              <FeatureCard key={card.title} {...card} />
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
