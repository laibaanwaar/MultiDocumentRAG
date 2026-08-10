import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthMessage from '../../components/auth/AuthMessage';
import { getSubscriptions } from '../../services/adminService';
import { clearAuthSession, isAuthFailure, parseAuthApiError } from '../../services/authService';

function formatDate(value) {
  if (!value) {
    return 'N/A';
  }

  try {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    }).format(new Date(value));
  } catch {
    return 'N/A';
  }
}

function formatAmount(value) {
  if (value == null || value === '') {
    return 'Rs. 0';
  }

  const numericValue = Number(value);
  return `Rs. ${numericValue.toLocaleString('en-US', {
    minimumFractionDigits: numericValue % 1 === 0 ? 0 : 2,
    maximumFractionDigits: 2
  })}`;
}

function formatBillingPeriod(value) {
  if (!value) {
    return 'N/A';
  }

  if (value === 'monthly') {
    return 'Monthly';
  }

  if (value === 'yearly') {
    return 'Yearly';
  }

  return value;
}

function statusPillClass(status) {
  if (status === 'active') {
    return 'bg-emerald-50 text-emerald-700';
  }

  return 'bg-amber-50 text-amber-700';
}

function planPillClass(code) {
  if (code === 'PRO') {
    return 'bg-violet-50 text-violet-700';
  }

  if (code === 'BASIC') {
    return 'bg-emerald-50 text-emerald-700';
  }

  return 'bg-slate-100 text-slate-600';
}

function displayUserName(subscription) {
  return subscription.user?.username || '—';
}

function displayUserEmail(subscription) {
  return subscription.user?.email || '—';
}

function getInitials(value) {
  const text = (value || '').trim();

  if (!text) {
    return 'U';
  }

  const parts = text.split(/\s+/);
  if (parts.length === 1) {
    return parts[0].slice(0, 2).toUpperCase();
  }

  return `${parts[0][0] || ''}${parts[1][0] || ''}`.toUpperCase();
}

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4 flex-none text-slate-400">
      <circle cx="11" cy="11" r="6.5" stroke="currentColor" strokeWidth="1.7" />
      <path d="m16 16 4 4" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4 flex-none text-slate-400">
      <rect x="4" y="5" width="16" height="15" rx="2" stroke="currentColor" strokeWidth="1.7" />
      <path d="M8 3.5v3M16 3.5v3M4 9.5h16" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
    </svg>
  );
}

function MoreIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4 flex-none text-slate-500">
      <circle cx="12" cy="5" r="1.6" fill="currentColor" />
      <circle cx="12" cy="12" r="1.6" fill="currentColor" />
      <circle cx="12" cy="19" r="1.6" fill="currentColor" />
    </svg>
  );
}

function PaginationArrow({ direction }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true" className="h-4 w-4 flex-none">
      {direction === 'left' ? (
        <path d="M14.5 6 8.5 12l6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      ) : (
        <path d="M9.5 6 15.5 12l-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      )}
    </svg>
  );
}

function LoadingRows() {
  return Array.from({ length: 5 }).map((_, index) => (
    <tr key={index} className="border-b border-[#f5efe6] last:border-b-0">
      {Array.from({ length: 8 }).map((__, cellIndex) => (
        <td key={cellIndex} className="px-5 py-4">
          <div className="h-8 animate-pulse rounded bg-slate-100" />
        </td>
      ))}
    </tr>
  ));
}

export default function AdminSubscriptions() {
  const navigate = useNavigate();
  const lastRequestId = useRef(0);
  const hasSuccessfulLoad = useRef(false);
  const [subscriptions, setSubscriptions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [count, setCount] = useState(0);
  const [page, setPage] = useState(1);
  const [nextPage, setNextPage] = useState(null);
  const [previousPage, setPreviousPage] = useState(null);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [selectedPlan, setSelectedPlan] = useState('');
  const [selectedStatus, setSelectedStatus] = useState('');

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setDebouncedSearch(search.trim());
      setPage((current) => (current === 1 ? current : 1));
    }, 400);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [search]);

  useEffect(() => {
    const currentRequestId = lastRequestId.current + 1;
    lastRequestId.current = currentRequestId;

    async function fetchSubscriptions() {
      setLoading(true);
      setError('');

      try {
        const response = await getSubscriptions(
          {
            page,
            search: debouncedSearch,
            plan: selectedPlan,
            status: selectedStatus
          }
        );

        if (lastRequestId.current !== currentRequestId) {
          return;
        }

        setSubscriptions(response?.results || []);
        setCount(response?.count || 0);
        setNextPage(response?.next || null);
        setPreviousPage(response?.previous || null);
        hasSuccessfulLoad.current = true;
      } catch (apiError) {
        const parsedError = parseAuthApiError(apiError, {}, { throttleMessage: 'Too many requests. Please wait a moment and try again.' });

        if (isAuthFailure(parsedError)) {
          clearAuthSession();
          navigate('/login', { replace: true });
          return;
        }

        if (parsedError.status === 403) {
          navigate('/chatbot', { replace: true });
          return;
        }

        if (lastRequestId.current !== currentRequestId) {
          return;
        }

        setError(parsedError.message || 'Unable to load subscriptions data.');

        if (!hasSuccessfulLoad.current) {
          setSubscriptions([]);
          setCount(0);
          setNextPage(null);
          setPreviousPage(null);
        }
      } finally {
        if (lastRequestId.current === currentRequestId) {
          setLoading(false);
        }
      }
    }

    fetchSubscriptions();
  }, [page, debouncedSearch, selectedPlan, selectedStatus, navigate]);

  const planOptions = useMemo(() => {
    const unique = new Map();

    subscriptions.forEach((item) => {
      if (item.plan?.code && item.plan?.name && !unique.has(item.plan.code)) {
        unique.set(item.plan.code, item.plan.name);
      }
    });

    return Array.from(unique.entries()).map(([code, name]) => ({ code, name }));
  }, [subscriptions]);

  function handlePlanChange(event) {
    setSelectedPlan(event.target.value);
    setPage(1);
  }

  function handleStatusChange(event) {
    setSelectedStatus(event.target.value);
    setPage(1);
  }

  function handleReset() {
    setSearch('');
    setDebouncedSearch('');
    setSelectedPlan('');
    setSelectedStatus('');
    setPage(1);
    setError('');
  }

  const showingFrom = count === 0 ? 0 : (page - 1) * subscriptions.length + 1;
  const showingTo = count === 0 ? 0 : showingFrom + subscriptions.length - 1;

  return (
    <section className="overflow-hidden rounded-[12px] border border-[#eadfce] bg-white shadow-[0_8px_18px_rgba(15,23,42,0.04)]">
      <div className="border-b border-[#f1eadf] px-6 py-5">
        <h2 className="text-[28px] font-serif font-bold leading-none text-[#111827]">Subscriptions</h2>
        <p className="mt-2 text-[14px] text-slate-500">Manage user subscriptions and their status.</p>
      </div>

      <div className="px-6 py-5">
        {error ? <AuthMessage variant="error">{error}</AuthMessage> : null}

        <div className="mt-4 grid gap-3 lg:grid-cols-[minmax(0,1.5fr)_180px_180px_150px_150px_92px]">
          <label className="flex h-11 min-w-0 items-center gap-2 rounded-[10px] border border-[#e3d8c7] bg-white px-3">
            <SearchIcon />
            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search by username or email..."
              style={{ color: '#111827' }}
              className="w-full min-w-0 bg-transparent text-[14px] !text-black outline-none placeholder:!text-black"
            />
          </label>

          <select
            value={selectedPlan}
            onChange={handlePlanChange}
            style={{ color: '#111827' }}
            className="h-11 rounded-[10px] border border-[#e3d8c7] bg-white px-3 text-[14px] !text-black outline-none"
          >
            <option value="">All Plans</option>
            {planOptions.map((plan) => (
              <option key={plan.code} value={plan.code}>
                {plan.name}
              </option>
            ))}
          </select>

          <select
            value={selectedStatus}
            onChange={handleStatusChange}
            style={{ color: '#111827' }}
            className="h-11 rounded-[10px] border border-[#e3d8c7] bg-white px-3 text-[14px] !text-black outline-none"
          >
            <option value="">All Status</option>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
            <option value="cancelled">Cancelled</option>
            <option value="expired">Expired</option>
          </select>

          <label className="flex h-11 items-center gap-2 rounded-[10px] border border-[#eee7db] bg-[#faf9f6] px-3">
            <CalendarIcon />
            <input
              type="text"
              disabled
              placeholder="Start Date"
              style={{ color: '#111827' }}
              className="w-full bg-transparent text-[14px] !text-black outline-none placeholder:!text-black disabled:!text-black disabled:opacity-100"
            />
          </label>

          <label className="flex h-11 items-center gap-2 rounded-[10px] border border-[#eee7db] bg-[#faf9f6] px-3">
            <CalendarIcon />
            <input
              type="text"
              disabled
              placeholder="End Date"
              style={{ color: '#111827' }}
              className="w-full bg-transparent text-[14px] !text-black outline-none placeholder:!text-black disabled:!text-black disabled:opacity-100"
            />
          </label>

          <button
            type="button"
            onClick={handleReset}
            className="h-11 rounded-[10px] border border-[#d8c5aa] bg-white px-4 text-[14px] font-semibold text-[#c39241] transition hover:bg-[#fbf7f1]"
          >
            Reset
          </button>
        </div>

        <div className="mt-5 overflow-hidden rounded-[12px] border border-[#eee7db] bg-white">
          <div className="overflow-x-auto">
            <table className="min-w-[1080px] w-full border-collapse text-left">
              <thead>
                <tr className="border-b border-[#f1eadf] text-[12px] font-semibold text-slate-600">
                  <th className="px-5 py-4">User</th>
                  <th className="px-5 py-4">Plan</th>
                  <th className="px-5 py-4">Amount</th>
                  <th className="px-5 py-4">Billing Cycle</th>
                  <th className="px-5 py-4">Start Date</th>
                  <th className="px-5 py-4">Next Billing</th>
                  <th className="px-5 py-4">Status</th>
                  <th className="px-5 py-4 text-center">Actions</th>
                </tr>
              </thead>

              <tbody className="text-[14px] text-slate-700">
                {loading ? <LoadingRows /> : null}

                {!loading && subscriptions.length === 0 ? (
                  <tr>
                    <td colSpan={8} className="px-5 py-10 text-center text-slate-500">
                      No data available
                    </td>
                  </tr>
                ) : null}

                {!loading
                  ? subscriptions.map((subscription) => (
                      <tr key={subscription.id} className="border-b border-[#f5efe6] last:border-b-0">
                        <td className="px-5 py-4">
                          <div className="flex items-center gap-3">
                            <div className="flex h-10 w-10 flex-none items-center justify-center rounded-full bg-[#f4efe6] text-[12px] font-semibold text-[#8b6c3d]">
                              {getInitials(displayUserName(subscription))}
                            </div>
                            <div className="min-w-0">
                              <p className="truncate text-[14px] font-semibold text-[#111827]">{displayUserName(subscription)}</p>
                              <p className="mt-0.5 truncate text-[12px] text-slate-500">{displayUserEmail(subscription)}</p>
                            </div>
                          </div>
                        </td>

                        <td className="px-5 py-4">
                          <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${planPillClass(subscription.plan?.code)}`}>
                            {subscription.plan?.name || '—'}
                          </span>
                        </td>

                        <td className="px-5 py-4 font-semibold text-[#1f2937]">{formatAmount(subscription.plan?.price)}</td>
                        <td className="px-5 py-4 text-slate-700">{formatBillingPeriod(subscription.plan?.billing_period)}</td>
                        <td className="px-5 py-4 text-slate-700">{formatDate(subscription.started_at)}</td>
                        <td className="px-5 py-4 text-slate-700">{formatDate(subscription.expires_at)}</td>

                        <td className="px-5 py-4">
                          <span className={`inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusPillClass(subscription.status)}`}>
                            {subscription.status ? subscription.status.charAt(0).toUpperCase() + subscription.status.slice(1) : '—'}
                          </span>
                        </td>

                        <td className="px-5 py-4 text-center">
                          <button
                            type="button"
                            className="inline-flex h-8 w-8 items-center justify-center rounded-full transition hover:bg-slate-50"
                          >
                            <MoreIcon />
                          </button>
                        </td>
                      </tr>
                    ))
                  : null}
              </tbody>
            </table>
          </div>
        </div>

        <div className="mt-5 flex flex-col gap-3 text-[13px] text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p>
            Showing {showingFrom} to {showingTo} of {count} results
          </p>

          <div className="flex items-center gap-2">
            <button
              type="button"
              disabled={!previousPage || loading}
              onClick={() => setPage((current) => Math.max(1, current - 1))}
              className="flex h-8 w-8 items-center justify-center rounded-[8px] border border-[#d8c5aa] text-slate-700 transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              <PaginationArrow direction="left" />
            </button>

            <span className="flex h-8 min-w-[34px] items-center justify-center rounded-[8px] border border-[#d8c5aa] px-3 text-[12px] font-semibold text-slate-700">
              {page}
            </span>

            <button
              type="button"
              disabled={!nextPage || loading}
              onClick={() => setPage((current) => current + 1)}
              className="flex h-8 w-8 items-center justify-center rounded-[8px] border border-[#d8c5aa] text-slate-700 transition disabled:cursor-not-allowed disabled:opacity-50"
            >
              <PaginationArrow direction="right" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}
