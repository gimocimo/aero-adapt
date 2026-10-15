import { expect, test } from "@playwright/test";

// Live mode end to end (M2 spec D10): Skyview against `aeroadapt serve` (started by the
// Playwright config on port 8799).
test("live mode: a fault injected from the panel is logged and detected; a sensor fault clears", async ({ page }) => {
  const errors: string[] = [];
  const messages: string[] = [];
  page.on("console", (msg) => {
    messages.push(`${msg.type()}: ${msg.text()}`);
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/?live=ws://localhost:8799&tiles=0");
  await expect(page.getByTestId("clock")).toContainText("LIVE", { timeout: 60_000 });
  await page.getByRole("button", { name: "4×" }).click();

  // Retried: a click that lands just before a dev-server reload is lost with the page.
  const active = page.getByTestId("active-faults");
  try {
    await expect(async () => {
      await page.getByRole("button", { name: "Motor cut", exact: true }).click();
      await expect(active).toContainText("Motor cut @", { timeout: 3_000 });
    }).toPass({ timeout: 30_000 });
  } catch (e) {
    // Diagnostics for CI: what the client last heard from the server.
    const state = await page.evaluate(() => {
      const live = (window as unknown as { __store?: { getState(): { live: unknown } } }).__store?.getState().live;
      return JSON.stringify(live, (k, v: unknown) => (k === "airframes" ? undefined : v));
    });
    console.log(`live state: ${state}\nconsole:\n${messages.slice(-30).join("\n")}`);
    throw e;
  }
  await expect(page.getByText("Fault: motor cut")).toBeVisible({ timeout: 20_000 });
  await expect(page.getByText("Motor failure detected")).toBeVisible({ timeout: 30_000 });

  await page.locator("summary", { hasText: "GNSS" }).click();
  await page.getByRole("button", { name: "GNSS dropout", exact: true }).click();
  await expect(active).toContainText("GNSS dropout @");
  await active.getByRole("button", { name: "clear" }).click();
  await expect(active).toContainText("cleared");
  await expect(page.getByText("Fault cleared: gnss dropout")).toBeVisible({ timeout: 20_000 });

  expect(errors).toEqual([]);
});
