import React, { useRef } from 'react';
import { UploadCloud, Image as ImageIcon, Loader2, Sparkles } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Progress } from '@/components/ui/progress';

export function ImageSearchZone({ onImageSelected, isLoading, workerStatus, uploadProgress, fileBlob }) {
    const inputRef = useRef(null);

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            onImageSelected(e.target.files[0]);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
            onImageSelected(files[0]);
        }
    };

    return (
        <div className="w-full">
            <div
                className={cn(
                    "relative overflow-hidden group border-2 border-dashed rounded-3xl p-8 sm:p-12 transition-all duration-300 cursor-pointer bg-white/50 backdrop-blur shadow-sm hover:shadow-md",
                    "border-slate-300 hover:border-rose-400 hover:bg-slate-50",
                    isLoading || workerStatus === 'loading' ? "pointer-events-none opacity-80" : ""
                )}
                onClick={() => inputRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    ref={inputRef}
                    onChange={handleFileChange}
                    className="hidden"
                    accept="image/jpeg, image/png, image/webp"
                />

                <div className="flex flex-col items-center justify-center space-y-4 text-center">
                    {fileBlob ? (
                        <div className="relative w-32 h-32 sm:w-48 sm:h-48 rounded-xl overflow-hidden shadow-lg border border-slate-200 mb-2 group-hover:scale-105 transition-transform mx-auto">
                            <img src={URL.createObjectURL(fileBlob)} className="w-full h-full object-cover" alt="Search query" />
                            {isLoading && (
                                <div className="absolute inset-0 bg-white/60 flex items-center justify-center backdrop-blur-sm">
                                    <Loader2 className="w-8 h-8 text-rose-600 animate-spin" />
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className={cn(
                            "p-4 rounded-full transition-all duration-500",
                            "bg-slate-100 text-slate-500 group-hover:bg-rose-50 group-hover:text-rose-500 group-hover:scale-110"
                        )}>
                            {workerStatus === 'loading' || isLoading ? (
                                <Loader2 className="w-8 h-8 sm:w-12 sm:h-12 animate-spin" />
                            ) : (
                                <ImageIcon className="w-8 h-8 sm:w-12 sm:h-12" />
                            )}
                        </div>
                    )}

                    {!fileBlob && (
                        <div className="space-y-1">
                            <h3 className="text-xl sm:text-2xl font-bold text-slate-900">
                                Glissez une image ici
                            </h3>
                            <p className="text-sm text-slate-500">
                                Ou cliquez pour parcourir vos dossiers
                            </p>
                        </div>
                    )}

                    {workerStatus === 'loading' && (
                        <div className="w-full max-w-xs mx-auto mt-6 space-y-2">
                            <div className="flex items-center justify-between text-xs font-bold text-rose-600 uppercase tracking-widest">
                                <span>Chargement SigLIP</span>
                                <span>{Math.round(uploadProgress)}%</span>
                            </div>
                            <Progress value={uploadProgress} className="h-2 bg-slate-100" />
                        </div>
                    )}
                </div>
            </div>

            <div className="mt-4 flex flex-col sm:flex-row items-center justify-center gap-2 text-xs font-bold text-slate-500 bg-slate-50 py-2 px-4 rounded-xl w-fit mx-auto border border-slate-200 shadow-sm">
                <div className="flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-rose-500" /> Analyse IA (WebGPU)</div>
                <span className="hidden sm:inline">•</span>
                <div className="text-slate-400 font-medium">100% locale, privée et rapide</div>
            </div>
        </div>
    );
}
