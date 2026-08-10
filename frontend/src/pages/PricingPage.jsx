import PricingCard from '../components/pricing/PricingCard';
import SiteHeader from '../components/layout/SiteHeader';

const plans = [
  {
    name: 'Free',
    subtitle: 'For getting started',
    price: 'Rs. 0',
    buttonLabel: 'Get Started',
    features: ['5 AI chats per day', 'Access to public laws', 'Basic legal summaries']
  },
  {
    name: 'Basic',
    subtitle: 'For regular researchers',
    price: 'Rs. 999',
    buttonLabel: 'Start Basic Plan',
    features: [
      'Unlimited AI chats',
      'Access to Pakistani laws & acts',
      'Advanced summaries & insights',
      'Export & copy responses'
    ],
    highlighted: true
  },
  {
    name: 'Pro',
    subtitle: 'For legal professionals',
    price: 'Rs. 2,499',
    buttonLabel: 'Start Pro Plan',
    features: [
      'Everything in Basic',
      'Citations & legal references',
      'Priority support',
      'Early access to new features'
    ]
  }
];

export default function PricingPage() {
  return (
    <>
      <SiteHeader activeItem="Pricing" />

      <main className="min-h-[calc(100vh-72px)] bg-[radial-gradient(circle_at_top,rgba(216,168,84,0.08),transparent_18%),linear-gradient(180deg,#06111d_0%,#07121d_100%)] px-4 py-8 sm:px-6 lg:px-10 lg:py-10">
        <section className="mx-auto max-w-[1120px]">
          <div className="mx-auto max-w-[760px] text-center">
            <div className="inline-flex items-center rounded-full border border-[#a98241]/28 bg-[#0c1724] px-4 py-2 text-[13px] font-medium text-[#dcb56a]">
              Simple · Transparent · Built for Pakistan Law
            </div>

            <h1 className="mt-5 font-serif text-[clamp(2.3rem,4vw,4rem)] font-semibold leading-[1.02] tracking-tight text-[#f4efe6]">
              Simple pricing for <span className="text-[#d6a655]">legal research</span>
            </h1>

            <p className="mt-4 text-[16px] text-slate-400">
              Choose the plan that fits your needs. Upgrade or downgrade anytime.
            </p>
          </div>

          <div className="mt-10 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
            {plans.map((plan) => (
              <PricingCard key={plan.name} {...plan} />
            ))}
          </div>
        </section>
      </main>
    </>
  );
}
