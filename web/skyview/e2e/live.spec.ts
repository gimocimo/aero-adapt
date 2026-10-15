import { expect, type Page, test } from "@playwright/test";

// Live mode end to end (M2 spec D10): Skyview against `aeroadapt serve` (started by the
// Playwright config on port 8799). A sensor fault is injected and cleared in normal flight,
// then a motor cut is injected, logged and detected.

/** Click a fault button until the active-faults list shows it (a click that lands just
 * before a dev-server reload is lost with the page). */
async function inject(page: Page, group: string, name: string): Promise<void> {
  const button = page.getByRole("button", { name, exact: true });
  await expect(async () => {
    if (!(await button.isVisible())) await page.locator("summary", { hasText: group }).click();
    await button.click();
    await expect(page.getByTestId("active-faults")).toContainText(`${name} @`, { timeout: 3_000 });
  }).toPass({ timeout: 30_000 });
}

test("live mode: a sensor fault is injected and cleared; a motor cut is logged and detected", async ({ page }) => {
  const errors: string[] = [];
  const messages: string[] = [];
  page.on("console", (msg) => {
    messages.push(`${msg.type()}: ${msg.text()}`);
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/?live=ws://localhost:8799&tiles=0");
  await expect(page.getByTestId("clock")).toContainText("LIVE", { timeout: 60_000 });
  const active = page.getByTestId("active-faults");
  const log = page.locator(".eventlog"); // the HUD also badges active faults
  try {
    await inject(page, "GNSS", "GNSS dropout");
    await active.getByRole("button", { name: "clear" }).click();
    await expect(active).toContainText("cleared", { timeout: 10_000 });
    await expect(log.getByText("Fault cleared: gnss dropout")).toBeVisible({ timeout: 20_000 });

    await page.getByRole("button", { name: "2×" }).click();
    await inject(page, "Propulsion", "Motor cut");
    await expect(log.getByText("Fault: motor cut")).toBeVisible({ timeout: 20_000 });
    await expect(log.getByText("Motor failure detected")).toBeVisible({ timeout: 30_000 });
  } catch (e) {
    // Diagnostics for CI: what the client last heard from the server.
    const state = await page.evaluate(() => {
      const live = (window as unknown as { __store?: { getState(): { live: unknown } } }).__store?.getState().live;
      return JSON.stringify(live, (k, v: unknown) => (k === "airframes" ? undefined : v));
    });
    console.log(`live state: ${state}\nconsole:\n${messages.slice(-30).join("\n")}`);
    throw e;
  }
  expect(errors).toEqual([]);
});
