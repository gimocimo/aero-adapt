// Timings for loading a world, as performance measures ("skyview: …"), readable in the
// browser's performance panel or with performance.getEntriesByType("measure").
export function timed<T>(name: string, f: () => T): T {
  const t0 = performance.now();
  try {
    return f();
  } finally {
    performance.measure(name, { start: t0 });
  }
}
