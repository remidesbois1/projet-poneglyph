import { createDeepSeekOcrHandler } from '@/lib/server/deepseekOcr';

export const runtime = 'nodejs';
export const maxDuration = 130;
export const POST = createDeepSeekOcrHandler();
