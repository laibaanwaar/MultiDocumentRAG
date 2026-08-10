import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthButton from '../components/auth/AuthButton';
import AuthField from '../components/auth/AuthField';
import AuthLayout from '../components/auth/AuthLayout';
import AuthMessage from '../components/auth/AuthMessage';
import PasswordInput from '../components/auth/PasswordInput';
import {
  clearAuthSession,
  loginUser,
  parseAuthApiError,
  storeAuthSession
} from '../services/authService';

const emptyFieldErrors = {
  identifier: '',
  password: ''
};

export default function Login() {
  const location = useLocation();
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    identifier: location.state?.identifier || '',
    password: ''
  });
  const [fieldErrors, setFieldErrors] = useState(emptyFieldErrors);
  const [apiError, setApiError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleLoginSuccess(response) {
    storeAuthSession(response?.data);
    setSuccessMessage(response?.message || 'Login successful.');
    navigate('/chatbot', { replace: true });
  }

  function handleChange(event) {
    const { name, value } = event.target;
    setFormData((current) => ({
      ...current,
      [name]: value
    }));
    setFieldErrors((current) => ({
      ...current,
      [name]: ''
    }));
    setApiError('');
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (isSubmitting) {
      return;
    }

    const trimmedIdentifier = formData.identifier.trim();
    const nextFieldErrors = { ...emptyFieldErrors };

    if (!trimmedIdentifier) {
      nextFieldErrors.identifier = 'Username or email is required.';
    }

    if (!formData.password) {
      nextFieldErrors.password = 'Password is required.';
    }

    if (Object.values(nextFieldErrors).some(Boolean)) {
      setFieldErrors(nextFieldErrors);
      return;
    }

    setIsSubmitting(true);
    setApiError('');
    setSuccessMessage('');

    try {
      const response = await loginUser({
        identifier: trimmedIdentifier,
        password: formData.password
      });

      handleLoginSuccess(response);
    } catch (error) {
      const parsedError = parseAuthApiError(error, {
        identifier: 'identifier',
        password: 'password'
      });
      setFieldErrors((current) => ({
        ...current,
        ...parsedError.fieldErrors
      }));
      setApiError(parsedError.message);
      if (parsedError.status === 401) {
        clearAuthSession();
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <div className="mx-auto max-w-[258px]">
        <div className="mb-4 text-center">
          <h1 className="font-serif text-[18px] leading-none text-[#f5efe7]">Welcome Back</h1>
          <p className="mt-2 text-[10px] text-slate-400">
            Sign in to continue your legal research journey.
          </p>
        </div>

        <form className="space-y-2" onSubmit={handleSubmit}>
          <AuthField
            label="Username or Email"
            name="identifier"
            value={formData.identifier}
            placeholder="Enter your username or email"
            onChange={handleChange}
            icon="mail"
            error={fieldErrors.identifier}
            autoComplete="username"
          />
          <PasswordInput
            label="Password"
            name="password"
            value={formData.password}
            placeholder="Enter your password"
            onChange={handleChange}
            error={fieldErrors.password}
            autoComplete="current-password"
          />

          <AuthMessage variant="error">{apiError}</AuthMessage>
          <AuthMessage variant="success">{successMessage}</AuthMessage>

          <AuthButton type="submit" disabled={isSubmitting}>
            {isSubmitting ? 'Signing In...' : 'Log In'}
          </AuthButton>

          <p className="pt-0.5 text-center text-[10px] text-slate-400">
            Need an account?{' '}
            <button
              type="button"
              onClick={() => navigate('/signup')}
              className="font-semibold text-amber-300"
            >
              Create one
            </button>
          </p>
        </form>
      </div>
    </AuthLayout>
  );
}
