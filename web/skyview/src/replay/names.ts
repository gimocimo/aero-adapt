/** A fault bit's display name: "sensor:gnss_dropout" → "gnss dropout". */
export function faultName(bit: string): string {
  return (bit.split(":").at(-1) ?? bit).replaceAll("_", " ");
}
