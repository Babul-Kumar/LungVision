import React, { useState, useMemo } from 'react';
import { X, Search, BookOpen } from 'lucide-react';
import { MEDICAL_GLOSSARY } from '@/types/explainability';
import { Badge } from '@/components/common/Badge';

interface MedicalGlossaryModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const MedicalGlossaryModal: React.FC<MedicalGlossaryModalProps> = ({ isOpen, onClose }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<'all' | 'clinical' | 'technical' | 'regulatory'>('all');

  const filteredTerms = useMemo(() => {
    return MEDICAL_GLOSSARY.filter((term) => {
      const matchesSearch =
        term.term.toLowerCase().includes(searchQuery.toLowerCase()) ||
        term.shortDefinition.toLowerCase().includes(searchQuery.toLowerCase()) ||
        term.detailedExplanation.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || term.category === selectedCategory;
      return matchesSearch && matchesCategory;
    });
  }, [searchQuery, selectedCategory]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-medical-card border border-medical-border rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-medical-border flex items-center justify-between bg-medical-surface/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-teal-950/60 border border-teal-800/60 flex items-center justify-center text-medical-teal">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">Medical & AI Radiology Glossary</h3>
                <Badge variant="teal" size="sm">Plain-Language Reference</Badge>
              </div>
              <p className="text-xs text-medical-muted mt-0.5">
                Clear definitions of thoracic anatomy, radiographic terms, and deep learning metrics.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-medical-muted hover:text-white hover:bg-medical-surface transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="p-4 bg-medical-surface/40 border-b border-medical-border/80 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-medical-muted absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search medical or technical terms (e.g., Atelectasis, AUROC, Threshold)..."
              className="w-full bg-medical-surface border border-medical-border rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-medical-muted focus:outline-none focus:border-medical-teal"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            {(['all', 'clinical', 'technical'] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg capitalize transition-all ${
                  selectedCategory === cat
                    ? 'bg-medical-teal text-slate-950 font-semibold shadow-sm'
                    : 'bg-medical-surface text-medical-muted hover:text-white border border-medical-border'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Glossary Terms List */}
        <div className="p-5 overflow-y-auto space-y-3.5">
          {filteredTerms.length === 0 ? (
            <div className="text-center py-12 text-medical-muted text-xs">
              No matching glossary terms found for "{searchQuery}".
            </div>
          ) : (
            filteredTerms.map((item) => (
              <div
                key={item.term}
                className="p-4 rounded-xl bg-medical-surface border border-medical-border/80 space-y-2 hover:border-medical-teal/40 transition-colors"
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    {item.term}
                  </h4>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-medical-card border border-medical-border text-medical-teal">
                    {item.category}
                  </span>
                </div>
                <p className="text-xs font-medium text-slate-200">{item.shortDefinition}</p>
                <p className="text-xs text-medical-muted leading-relaxed pt-1 border-t border-medical-border/40">
                  {item.detailedExplanation}
                </p>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-medical-border bg-medical-surface/80 flex items-center justify-between text-xs text-medical-muted">
          <span>{filteredTerms.length} terms displayed</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-medical-surface border border-medical-border text-white hover:bg-medical-card transition-colors text-xs font-medium"
          >
            Close Glossary
          </button>
        </div>
      </div>
    </div>
  );
};
