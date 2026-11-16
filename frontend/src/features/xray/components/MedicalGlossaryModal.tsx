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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-[#1a082e]/50 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-white border border-[#e2e8f0] rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-[#e2e8f0] flex items-center justify-between bg-slate-50/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-50 border border-amber-200 flex items-center justify-center text-amber-700 shadow-xs">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-[#1a082e]">Medical & AI Radiology Glossary</h3>
                <Badge variant="amber" size="sm">Plain-Language Reference</Badge>
              </div>
              <p className="text-xs text-[#5e4b75] mt-0.5">
                Clear definitions of thoracic anatomy, radiographic terms, and deep learning metrics.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-[#5e4b75] hover:text-[#1a082e] hover:bg-slate-200/50 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="p-4 bg-slate-100/70 border-b border-[#e2e8f0] flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-[#5e4b75] absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search medical or technical terms (e.g., Atelectasis, AUROC, Threshold)..."
              className="w-full bg-white border border-[#cbd5e1] rounded-lg pl-9 pr-3 py-1.5 text-xs text-[#1a082e] placeholder-[#5e4b75]/70 focus:outline-none focus:border-amber-500 shadow-xs"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            {(['all', 'clinical', 'technical'] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg capitalize transition-all font-medium ${
                  selectedCategory === cat
                    ? 'bg-gradient-to-r from-amber-500 to-amber-600 text-white font-bold shadow-xs border border-amber-600'
                    : 'bg-white text-[#5e4b75] hover:text-[#1a082e] border border-[#e2e8f0]'
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
            <div className="text-center py-12 text-[#5e4b75] text-xs">
              No matching glossary terms found for "{searchQuery}".
            </div>
          ) : (
            filteredTerms.map((item) => (
              <div
                key={item.term}
                className="p-4 rounded-xl bg-white border border-[#e2e8f0] space-y-2 hover:border-amber-400/60 shadow-xs transition-colors"
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className="text-sm font-bold text-[#1a082e] flex items-center gap-2">
                    {item.term}
                  </h4>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-900 font-semibold">
                    {item.category}
                  </span>
                </div>
                <p className="text-xs font-medium text-[#1a082e]">{item.shortDefinition}</p>
                <p className="text-xs text-[#5e4b75] leading-relaxed pt-1 border-t border-[#e2e8f0]">
                  {item.detailedExplanation}
                </p>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-[#e2e8f0] bg-slate-50/90 flex items-center justify-between text-xs text-[#5e4b75]">
          <span>{filteredTerms.length} terms displayed</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-white border border-[#e2e8f0] text-[#1a082e] hover:border-amber-400 hover:bg-slate-50 transition-colors text-xs font-semibold shadow-xs"
          >
            Close Glossary
          </button>
        </div>
      </div>
    </div>
  );
};
