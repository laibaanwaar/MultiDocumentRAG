import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import RecentUsersTable from '../../components/admin/RecentUsersTable';
import AuthMessage from '../../components/auth/AuthMessage';
import { getAdminUsers } from '../../services/adminService';
import { clearAuthSession, isAuthFailure, parseAuthApiError } from '../../services/authService';

export default function AdminUsers() {
  const navigate = useNavigate();
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    async function loadUsers() {
      setLoading(true);
      setError('');

      try {
        const response = await getAdminUsers();
        setUsers(response?.results || []);
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

    loadUsers();
  }, [navigate]);

  return (
    <div className="space-y-5">
      {error ? <AuthMessage variant="error">{error}</AuthMessage> : null}
      <RecentUsersTable users={users} loading={loading} />
    </div>
  );
}
