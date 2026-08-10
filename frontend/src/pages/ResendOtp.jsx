import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthButton from '../components/auth/AuthButton';
import AuthField from '../components/auth/AuthField';
import AuthLayout from '../components/auth/AuthLayout';
import AuthMessage from '../components/auth/AuthMessage';
import {
  AUTH_RESEND_COOLDOWN_SECONDS,
  getStoredAuthEmail,
  isValidEmail,
  parseAuthApiError,
  resendOtp,
  storeAuthEmail
} from '../services/authService';

export default function ResendOtp() {
  const navigate = useNavigate();
  const location = useLocation();
  const initialEmail = location.state?.email || getStoredAuthEmail();
  const [email, setEmail] = useState(initialEmail);
  const [emailError, setEmailError] = useState('');
  const [apiError, setApiError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');
  const [resendTimer, setResendTimer] = useState(0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (resendTimer <= 0) {
      return undefined;
    }

    const timerId = window.setTimeout(() => {
      setResendTimer((current) => current - 1);
    }, 1000);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [resendTimer]);

  function formatTimer(totalSeconds) {
    const minutes = String(Math.floor(totalSeconds / 60)).padStart(2, '0');
    const seconds = String(totalSeconds % 60).padStart(2, '0');
    return `${minutes}:${seconds}`;
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (isSubmitting || resendTimer > 0) {
      return;
    }

    const normalizedEmail = email.trim().toLowerCase();

    if (!normalizedEmail) {
      setEmailError('Registered email is required.');
      return;
    }

    if (!isValidEmail(normalizedEmail)) {
      setEmailError('Enter a valid email address.');
      return;
    }

    setIsSubmitting(true);
    setEmailError('');
    setApiError('');
    setSuccessMessage('');

    try {
      const response = await resendOtp({
        email: normalizedEmail
      });

      storeAuthEmail(normalizedEmail);
      setSuccessMessage(response?.message || 'OTP sent successfully.');
      setResendTimer(
        Number(response?.data?.otp_expires_in_seconds)
          ? Math.min(Number(response.data.otp_expires_in_seconds), AUTH_RESEND_COOLDOWN_SECONDS)
          : AUTH_RESEND_COOLDOWN_SECONDS
      );
    } catch (error) {
      const parsedError = parseAuthApiError(error, {
        email: 'email'
      });
      setEmailError(parsedError.fieldErrors.email || '');
      setApiError(parsedError.message);
      if (parsedError.retryAfterSeconds > 0) {
        setResendTimer(parsedError.retryAfterSeconds);
      }
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <div className="mx-auto max-w-[258px]">
        <div className="mb-4 text-center">
          <h1 className="font-serif text-[18px] leading-none text-[#f5efe7]">Need a New Code?</h1>
          <p className="mt-2 text-[10px] leading-5 text-slate-400">
            Resend OTP to your registered email address.
          </p>
        </div>

        <form className="space-y-3" onSubmit={handleSubmit}>
          <AuthField
            label="Registered Email"
            name="email"
            type="email"
            value={email}
            placeholder="Enter your registered email"
            onChange={(event) => {
              setEmail(event.target.value);
              setEmailError('');
              setApiError('');
            }}
            icon="mail"
            error={emailError}
            autoComplete="email"
          />

          <AuthMessage variant="error">{apiError}</AuthMessage>
          <AuthMessage variant="success">{successMessage}</AuthMessage>

          {resendTimer > 0 ? (
            <p className="text-center text-[10px] text-amber-200">
              Resend available in {formatTimer(resendTimer)}
            </p>
          ) : null}

          <AuthButton type="submit" disabled={isSubmitting || resendTimer > 0}>
            {isSubmitting ? 'Sending New OTP...' : 'Send New OTP'}
          </AuthButton>

          <div className="space-y-2 text-center text-[10px] text-slate-400">
            <p>
              Already have the code?{' '}
              <button
                type="button"
                onClick={() => navigate('/verify-otp', { state: { email: email.trim().toLowerCase() } })}
                className="font-semibold text-amber-300"
              >
                Verify OTP
              </button>
            </p>
            <p>
              <button
                type="button"
                onClick={() => navigate('/signup')}
                className="font-semibold text-amber-300"
              >
                Back to Sign Up
              </button>
            </p>
          </div>
        </form>
      </div>
    </AuthLayout>
  );
}
