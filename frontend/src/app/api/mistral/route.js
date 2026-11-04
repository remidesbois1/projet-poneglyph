import { createMistralOcrHandler } from '@/lib/server/mistralOcr';

export const runtime = 'nodejs';
export const maxDuration = 130;
export const POST = createMistralOcrHandler();
