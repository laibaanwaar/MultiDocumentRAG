import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AdminStatCard from '../../components/admin/AdminStatCard';
import RecentUsersTable from '../../components/admin/RecentUsersTable';
import SubscriptionOverview from '../../components/admin/SubscriptionOverview';
import AuthMessage from '../../components/auth/AuthMessage';
import { getAdminDashboard } from '../../services/adminService';
import { clearAuthSession, isAuthFailure, parseAuthApiError } from '../../services/authService';

export default function AdminDashboard() {
  const navigate = useNavigate();
  const [dashboard, setDashboard] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function loadDashboard() {
      setLoading(true);
      setError('');

      try {
        const response = await getAdminDashboard();
        setDashboard(response?.data || null);
      } catch (apiError) {
        const parsedError = parseAuthApiError(apiError);

        if (isAuthFailure(parsedError)) {
          clearAuthSession();
          navigate('/login', { replace: true });
          return;
        }

        if (parsedError.status === 403) {
          navigate('/chatbot', { replace: true });
          return;
        }

        setError(parsedError.status === 0 ? 'Unable to load dashboard data.' : parsedError.message || 'Unable to load dashboard data.');
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, [navigate]);

  const summary = dashboard?.summary;

  return (
    <div className="mx-auto max-w-[1220px] space-y-5">
      {error ? <AuthMessage variant="error">{error}</AuthMessage> : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        <AdminStatCard label="Total Users" value={summary?.total_users} loading={loading} />
        <AdminStatCard label="Paid Subscribers" value={summary?.paid_subscribers} loading={loading} />
        <AdminStatCard label="Total Active Subscribers" value={summary?.active_subscribers} loading={loading} />
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.7fr)_minmax(320px,0.95fr)]">
        <RecentUsersTable users={dashboard?.recent_users || []} loading={loading} />
        <SubscriptionOverview plans={dashboard?.subscription_overview || []} loading={loading} />
      </div>
    </div>
  );
}
