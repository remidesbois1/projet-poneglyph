import registry from '@poneglyph/shared/model-registry.json';

export const PUBLISHED_MODEL_BENCHMARKS = Object.entries(registry.models)
    .filter(([, model]) => model.benchmark?.status === 'published')
    .map(([id, model]) => ({ id, ...model }));

function getModelRegistryEntry(modelId) {
    return registry.models[modelId] || null;
}

export function formatRegistryMetric(modelId, metricKey = null) {
    const model = getModelRegistryEntry(modelId);
    if (!model || model.benchmark?.status !== 'published') return 'Benchmark comparable non publié';

    const key = metricKey || model.benchmark.primary_metric;
    const metric = model.benchmark.metrics[key];
    if (!metric) return 'Métrique non publiée';

    const numericValue = metric.unit === 'ratio' ? metric.value * 100 : metric.value;
    const value = numericValue.toFixed(metric.display_precision).replace('.', ',');
    return `${metric.label} ${value}${metric.unit === 'ratio' ? ' %' : ''}`;
}
