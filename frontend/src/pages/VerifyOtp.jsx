import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import AuthButton from '../components/auth/AuthButton';
import AuthLayout from '../components/auth/AuthLayout';
import AuthMessage from '../components/auth/AuthMessage';
import OtpInput from '../components/auth/OtpInput';
import {
  clearStoredAuthEmail,
  getStoredAuthEmail,
  parseAuthApiError,
  verifyOtp
} from '../services/authService';

export default function VerifyOtp() {
  const navigate = useNavigate();
  const location = useLocation();
  const emailFromState = location.state?.email || '';
  const signupMessage = location.state?.signupMessage || '';
  const [email] = useState(emailFromState || getStoredAuthEmail());
  const [otp, setOtp] = useState('');
  const [otpError, setOtpError] = useState('');
  const [apiError, setApiError] = useState('');
  const [successMessage, setSuccessMessage] = useState(signupMessage);
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    if (!email) {
      navigate('/signup', { replace: true });
    }
  }, [email, navigate]);

  async function handleSubmit(event) {
    event.preventDefault();

    if (isSubmitting) {
      return;
    }

    if (otp.length !== 6) {
      setOtpError('Enter the 6-digit verification code.');
      return;
    }

    setIsSubmitting(true);
    setOtpError('');
    setApiError('');
    setSuccessMessage('');

    try {
      const response = await verifyOtp({
        email,
        otp
      });

      setSuccessMessage(response?.message || 'Email verified successfully.');
      clearStoredAuthEmail();
      setTimeout(() => {
        navigate('/login', {
          replace: true,
          state: {
            identifier: email
          }
        });
      }, 900);
    } catch (error) {
      const parsedError = parseAuthApiError(error, {
        otp: 'otp'
      });
      setOtpError(parsedError.fieldErrors.otp || '');
      setApiError(parsedError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <div className="mx-auto max-w-[258px]">
        <div className="mb-4 text-center">
          <h1 className="font-serif text-[18px] leading-none text-[#f5efe7]">Verify Your Account</h1>
          <p className="mt-2 text-[10px] leading-5 text-slate-400">
            A 6-digit verification code was sent to your email.
          </p>
          <p className="mt-2 rounded-xl bg-white/5 px-3 py-2 text-[10px] font-medium text-amber-200">
            {email}
          </p>
        </div>

        <form className="space-y-3" onSubmit={handleSubmit}>
          <OtpInput value={otp} onChange={(nextOtp) => {
            setOtp(nextOtp);
            setOtpError('');
            setApiError('');
          }} error={otpError} />

          <AuthMessage variant="error">{apiError}</AuthMessage>
          <AuthMessage variant="success">{successMessage}</AuthMessage>

          <AuthButton type="submit" disabled={isSubmitting}>
            {isSubmitting ? 'Verifying OTP...' : 'Verify OTP'}
          </AuthButton>

          <div className="space-y-2 text-center text-[10px] text-slate-400">
            <p>
              Didn&apos;t receive the code?{' '}
              <button
                type="button"
                onClick={() => navigate('/resend-otp', { state: { email } })}
                className="font-semibold text-amber-300"
              >
                Resend OTP
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
