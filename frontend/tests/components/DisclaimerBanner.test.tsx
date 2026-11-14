import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { DisclaimerBanner } from '@/components/common/DisclaimerBanner';

describe('DisclaimerBanner Component', () => {
  it('renders research and non-diagnostic regulatory notice', () => {
    render(<DisclaimerBanner />);

    expect(screen.getByText('Research Decision-Support Disclaimer')).toBeInTheDocument();
    expect(screen.getByText(/experimental artificial intelligence platform/i)).toBeInTheDocument();
  });

  it('renders compact mode when compact prop is true', () => {
    render(<DisclaimerBanner compact />);

    expect(screen.getByText(/For research and educational use only/i)).toBeInTheDocument();
  });
});
