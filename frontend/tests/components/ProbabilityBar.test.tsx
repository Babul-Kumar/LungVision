import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { ProbabilityBar } from '@/components/common/ProbabilityBar';

describe('ProbabilityBar Component', () => {
  it('renders label and formatted probability percentage', () => {
    render(
      <ProbabilityBar
        label="Cardiomegaly"
        probability={0.8543}
        threshold={0.950195}
        experimentalFlag={false}
      />
    );

    expect(screen.getByText('Cardiomegaly')).toBeInTheDocument();
    expect(screen.getByText('85.43%')).toBeInTheDocument();
    expect(screen.getByText(/Thresh:/)).toBeInTheDocument();
    expect(screen.getByText('95.0%')).toBeInTheDocument();
  });

  it('renders Above Threshold badge when experimentalFlag is true', () => {
    render(
      <ProbabilityBar
        label="Pneumothorax"
        probability={0.65}
        threshold={0.58252}
        experimentalFlag={true}
      />
    );

    expect(screen.getByText('Pneumothorax')).toBeInTheDocument();
    expect(screen.getByText('Above Threshold')).toBeInTheDocument();
    expect(screen.getByText('65.00%')).toBeInTheDocument();
  });

  it('renders Negative status when below calibrated threshold', () => {
    render(
      <ProbabilityBar
        label="Edema"
        probability={0.12}
        threshold={0.876953}
        experimentalFlag={false}
      />
    );

    expect(screen.getByText('Negative')).toBeInTheDocument();
    expect(screen.getByText('12.00%')).toBeInTheDocument();
  });
});
