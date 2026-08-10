import { useRef } from 'react';

const OTP_LENGTH = 6;

export default function OtpInput({ value, onChange, error }) {
  const inputRefs = useRef([]);

  function focusIndex(index) {
    inputRefs.current[index]?.focus();
    inputRefs.current[index]?.select();
  }

  function handleInputChange(index, nextValue) {
    const digit = nextValue.replace(/\D/g, '').slice(-1);
    const nextOtp = value.padEnd(OTP_LENGTH, ' ').split('');
    nextOtp[index] = digit || '';
    const combined = nextOtp.join('').replace(/\s/g, '');
    onChange(combined);

    if (digit && index < OTP_LENGTH - 1) {
      focusIndex(index + 1);
    }
  }

  function handleKeyDown(index, event) {
    if (event.key === 'Backspace' && !value[index] && index > 0) {
      focusIndex(index - 1);
    }

    if (event.key === 'ArrowLeft' && index > 0) {
      focusIndex(index - 1);
    }

    if (event.key === 'ArrowRight' && index < OTP_LENGTH - 1) {
      focusIndex(index + 1);
    }
  }

  function handlePaste(event) {
    const pastedDigits = event.clipboardData.getData('text').replace(/\D/g, '').slice(0, OTP_LENGTH);

    if (!pastedDigits) {
      return;
    }

    event.preventDefault();
    onChange(pastedDigits);
    focusIndex(Math.min(pastedDigits.length - 1, OTP_LENGTH - 1));
  }

  return (
    <div>
      <div className="grid grid-cols-6 gap-2">
        {Array.from({ length: OTP_LENGTH }).map((_, index) => (
          <input
            key={index}
            ref={(element) => {
              inputRefs.current[index] = element;
            }}
            type="text"
            inputMode="numeric"
            pattern="[0-9]*"
            maxLength={1}
            value={value[index] || ''}
            onChange={(event) => handleInputChange(index, event.target.value)}
            onKeyDown={(event) => handleKeyDown(index, event)}
            onPaste={handlePaste}
            aria-label={`OTP digit ${index + 1}`}
            className={`h-11 rounded-xl bg-[#111d2b]/88 text-center text-[16px] text-black outline-none transition ${
              error ? 'ring-1 ring-rose-400/50' : 'focus:bg-[#142131]/96'
            }`}
          />
        ))}
      </div>
      {error ? <p className="mt-2 text-[8px] text-rose-300">{error}</p> : null}
    </div>
  );
}
