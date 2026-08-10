import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthButton from '../components/auth/AuthButton';
import AuthField from '../components/auth/AuthField';
import AuthLayout from '../components/auth/AuthLayout';
import AuthMessage from '../components/auth/AuthMessage';
import PasswordInput from '../components/auth/PasswordInput';
import {
  parseAuthApiError,
  signupUser,
  splitFullName,
  isValidEmail,
  storeAuthEmail
} from '../services/authService';

const initialFormData = {
  fullName: '',
  email: '',
  username: '',
  password: '',
  confirmPassword: '',
  termsAccepted: true
};

const emptyFieldErrors = {
  fullName: '',
  email: '',
  username: '',
  password: '',
  confirmPassword: '',
  termsAccepted: ''
};

const signupFieldMap = {
  first_name: 'fullName',
  last_name: 'fullName',
  email: 'email',
  username: 'username',
  password: 'password',
  password_confirm: 'confirmPassword'
};

function UserAccountGlyph() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <circle cx="12" cy="8" r="3.2" />
      <path d="M5.5 19a6.5 6.5 0 0 1 13 0" />
    </svg>
  );
}

export default function Signup() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState(initialFormData);
  const [fieldErrors, setFieldErrors] = useState(emptyFieldErrors);
  const [apiError, setApiError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  function handleChange(event) {
    const { name, value, type, checked } = event.target;

    setFormData((current) => ({
      ...current,
      [name]: type === 'checkbox' ? checked : value
    }));

    setApiError('');
    setFieldErrors((current) => ({
      ...current,
      [name]: ''
    }));
  }

  async function handleSubmit(event) {
    event.preventDefault();

    if (isSubmitting) {
      return;
    }

    const trimmedFullName = formData.fullName.trim();
    const trimmedUsername = formData.username.trim();
    const trimmedEmail = formData.email.trim().toLowerCase();
    const nextFieldErrors = { ...emptyFieldErrors };

    if (!trimmedFullName) {
      nextFieldErrors.fullName = 'Full Name is required.';
    }

    if (!trimmedUsername) {
      nextFieldErrors.username = 'Username is required.';
    }

    if (!trimmedEmail) {
      nextFieldErrors.email = 'Email is required.';
    } else if (!isValidEmail(trimmedEmail)) {
      nextFieldErrors.email = 'Enter a valid email address.';
    }

    if (!formData.password) {
      nextFieldErrors.password = 'Password is required.';
    } else if (formData.password.length < 12) {
      nextFieldErrors.password = 'Password must be at least 12 characters long.';
    }

    if (!formData.confirmPassword) {
      nextFieldErrors.confirmPassword = 'Confirm Password is required.';
    } else if (formData.password !== formData.confirmPassword) {
      nextFieldErrors.confirmPassword = 'Passwords do not match.';
    }

    if (!formData.termsAccepted) {
      nextFieldErrors.termsAccepted = 'You must agree to the Terms of Service and Privacy Policy.';
    }

    if (Object.values(nextFieldErrors).some(Boolean)) {
      setFieldErrors(nextFieldErrors);
      return;
    }

    const { first_name, last_name } = splitFullName(trimmedFullName);
    const payload = {
      username: trimmedUsername,
      email: trimmedEmail,
      first_name,
      last_name,
      password: formData.password,
      password_confirm: formData.confirmPassword
    };

    setIsSubmitting(true);
    setApiError('');
    setFieldErrors({ ...emptyFieldErrors });

    try {
      const response = await signupUser(payload);
      storeAuthEmail(trimmedEmail);
      navigate('/verify-otp', {
        state: {
          email: trimmedEmail,
          purpose: 'signup',
          signupMessage:
            response?.message || 'Account created. A verification code has been sent to your email.'
        }
      });
    } catch (error) {
      const parsedError = parseAuthApiError(error, signupFieldMap);
      setFieldErrors((current) => ({
        ...current,
        ...parsedError.fieldErrors
      }));
      setApiError(parsedError.message);
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <AuthLayout>
      <div className="mx-auto max-w-[258px]">
        <div className="mb-3 text-center">
          <div className="mx-auto mb-2.5 flex w-fit items-center gap-3 text-amber-200">
            <span className="flex h-8 w-8 items-center justify-center rounded-full bg-amber-100/8">
              <UserAccountGlyph />
            </span>
          </div>
          <h1 className="font-serif text-[18px] leading-none text-[#f5efe7]">Create Your Account</h1>
          <p className="mt-1.5 text-[10px] text-slate-400">
            Join <span className="font-semibold text-amber-300">PakLaw AI</span> and simplify your legal research.
          </p>
        </div>

        <form className="space-y-1.5" onSubmit={handleSubmit}>
          <AuthField
            label="Full Name"
            name="fullName"
            value={formData.fullName}
            placeholder="Enter your full name"
            onChange={handleChange}
            icon="user"
            error={fieldErrors.fullName}
            autoComplete="name"
          />
          <AuthField
            label="Email Address"
            name="email"
            type="email"
            value={formData.email}
            placeholder="Enter your email address"
            onChange={handleChange}
            icon="mail"
            error={fieldErrors.email}
            autoComplete="email"
          />
          <AuthField
            label="Username"
            name="username"
            value={formData.username}
            placeholder="Choose a username"
            onChange={handleChange}
            icon="username"
            error={fieldErrors.username}
            autoComplete="username"
          />
          <PasswordInput
            label="Password"
            name="password"
            value={formData.password}
            placeholder="Create a strong password"
            onChange={handleChange}
            error={fieldErrors.password}
            helper="At least 12 characters with letters, numbers & symbols"
            autoComplete="new-password"
          />
          <PasswordInput
            label="Confirm Password"
            name="confirmPassword"
            value={formData.confirmPassword}
            placeholder="Confirm your password"
            onChange={handleChange}
            error={fieldErrors.confirmPassword}
            autoComplete="new-password"
          />

          <label className="mt-1 flex items-center gap-2 text-[10px] text-slate-300">
            <input
              type="checkbox"
              name="termsAccepted"
              checked={formData.termsAccepted}
              onChange={handleChange}
              className="h-3 w-3 rounded bg-transparent accent-[#e1b86d]"
            />
            <span>
              I agree to the <span className="text-amber-300">Terms of Service</span> and{' '}
              <span className="text-amber-300">Privacy Policy</span>
            </span>
          </label>

          {fieldErrors.termsAccepted ? (
            <p className="text-[8px] text-rose-300">{fieldErrors.termsAccepted}</p>
          ) : null}

          <AuthMessage variant="error">{apiError}</AuthMessage>

          <AuthButton type="submit" disabled={isSubmitting}>
            <UserAccountGlyph />
            {isSubmitting ? 'Creating Account...' : 'Create Account'}
          </AuthButton>

          <p className="pt-0.5 text-center text-[10px] text-slate-400">
            Already have an account?{' '}
            <button
              type="button"
              onClick={() => navigate('/login')}
              className="font-semibold text-amber-300"
            >
              Log in
            </button>
          </p>
        </form>
      </div>
    </AuthLayout>
  );
}
