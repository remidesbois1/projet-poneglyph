import { Quote, ScanText, Sparkles } from 'lucide-react';

/** Each mode borrows the colour of a kind of Poneglyph. `theme` feeds the `--mode-*` variables the whole page is styled with. */
export const SEARCH_MODES = [
    {
        id: 'keyword',
        label: 'Textuel',
        Icon: Quote,
        stone: 'Poneglyph',
        tagline: 'retrouver les traces écrites et les paroles exactes.',
        placeholder: 'Cherchez un dialogue exact…',
        theme: {
            '--mode': '#8dbbff',
            '--mode-strong': '#3d86ff',
            '--mode-strong-hover': '#2f73dc',
            '--mode-ink': '#ffffff',
            '--mode-rgb': '61 134 255',
        },
    },
    {
        id: 'semantic',
        label: 'Sémantique',
        Icon: Sparkles,
        stone: 'Rio Poneglyph',
        tagline: 'déchiffrer l’histoire à travers les concepts et les souvenirs.',
        placeholder: 'Ex : première rencontre entre Luffy et Sanji…',
        theme: {
            '--mode': '#ff8f85',
            '--mode-strong': '#a11010',
            '--mode-strong-hover': '#820d0d',
            '--mode-ink': '#ffffff',
            '--mode-rgb': '214 48 40',
        },
    },
    {
        id: 'ocr',
        label: 'Visuelle',
        Icon: ScanText,
        stone: 'Poneglyph Doré',
        tagline: 'retrouver une page depuis une image ou un crop.',
        placeholder: '',
        theme: {
            '--mode': '#f6c945',
            '--mode-strong': '#f2c230',
            '--mode-strong-hover': '#f8d35c',
            '--mode-ink': '#1b1403',
            '--mode-rgb': '242 194 48',
        },
    },
];

export function getSearchMode(id) {
    return SEARCH_MODES.find(mode => mode.id === id) || SEARCH_MODES[0];
}

/** Same scenes as the landing page, so a visitor finds the examples they were shown there. */
export const SEMANTIC_EXAMPLES = {
    'one-piece': [
        'Luffy rencontre Zoro attaché à un poteau',
        'le combat entre Zoro et Mihawk',
        'Nami demande de l’aide à Luffy',
        'Sanji nourrit Gin au Baratie',
    ],
};
