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
      <div className="relative w-full max-w-3xl max-h-[85vh] bg-[#140a22] border border-white/10 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
        {/* Header */}
        <div className="p-5 border-b border-white/10 flex items-center justify-between bg-[#0e0719]/90">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-amber-950/60 border border-amber-800/60 flex items-center justify-center text-amber-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="text-lg font-bold text-white">Medical & AI Radiology Glossary</h3>
                <Badge variant="amber" size="sm">Plain-Language Reference</Badge>
              </div>
              <p className="text-xs text-slate-400 mt-0.5">
                Clear definitions of thoracic anatomy, radiographic terms, and deep learning metrics.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="p-4 bg-black/30 border-b border-white/10 flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search medical or technical terms (e.g., Atelectasis, AUROC, Threshold)..."
              className="w-full bg-[#180d28]/90 border border-white/10 rounded-lg pl-9 pr-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-amber-400"
            />
          </div>

          <div className="flex items-center gap-1.5 text-xs">
            {(['all', 'clinical', 'technical'] as const).map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg capitalize transition-all ${
                  selectedCategory === cat
                    ? 'bg-gradient-to-r from-amber-400 via-amber-500 to-amber-600 text-slate-950 font-bold shadow-md shadow-amber-950/40 border border-amber-300/40'
                    : 'bg-medical-surface text-slate-400 hover:text-white border border-white/10'
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
            <div className="text-center py-12 text-slate-400 text-xs">
              No matching glossary terms found for "{searchQuery}".
            </div>
          ) : (
            filteredTerms.map((item) => (
              <div
                key={item.term}
                className="p-4 rounded-xl bg-medical-surface border border-white/10 space-y-2 hover:border-amber-400/40 transition-colors"
              >
                <div className="flex items-center justify-between gap-2">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    {item.term}
                  </h4>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-medical-card border border-white/10 text-amber-400">
                    {item.category}
                  </span>
                </div>
                <p className="text-xs font-medium text-slate-200">{item.shortDefinition}</p>
                <p className="text-xs text-slate-400 leading-relaxed pt-1 border-t border-white/[0.08]">
                  {item.detailedExplanation}
                </p>
              </div>
            ))
          )}
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-white/10 bg-[#0e0719]/90 flex items-center justify-between text-xs text-slate-400">
          <span>{filteredTerms.length} terms displayed</span>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg bg-medical-surface border border-white/10 text-white hover:border-amber-400/40 hover:bg-medical-card transition-colors text-xs font-medium"
          >
            Close Glossary
          </button>
        </div>
      </div>
    </div>
  );
};
