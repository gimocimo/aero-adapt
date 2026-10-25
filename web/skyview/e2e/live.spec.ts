import { expect, type Page, test } from "@playwright/test";

// Live mode end to end (M2 spec D10): Skyview against `aeroadapt serve` (started by the
// Playwright config on port 8799), in the four-panel view over Wrenfold Vale (docs/07). A
// sensor fault is injected and cleared in normal flight, then a motor cut is injected, logged
// and detected.

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

  await page.goto("/?live=ws://localhost:8799&quality=low");
  await expect(page.getByTestId("clock")).toContainText("LIVE", { timeout: 60_000 });
  await expect(page.getByTestId("view-map")).toContainText("Wrenfold Vale");
  await expect(page.getByTestId("gauge-clearance")).toBeVisible();
  await page.getByTestId("faults-tab").click();
  const active = page.getByTestId("active-faults");
  const log = page.getByTestId("ticker"); // the latest events, whichever tab is open
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

// Txt2Fly (docs/06 T8): a typed plan is checked by the server, shown with its verdict; plain
// words (the stand-in translator here) become a plan that is confirmed and flown, and a request
// to change a safety rule is refused.
test("txt2fly: typed and spoken plans are checked, confirmed and flown", async ({ page }) => {
  // A scenario switch, three plans built and checked, one flown: past two minutes on CI's
  // software GPU since the planner reads the map (docs/06 §17).
  test.setTimeout(240_000);
  const errors: string[] = [];
  page.on("pageerror", (err) => errors.push(err.message));
  await page.goto("/?live=ws://localhost:8799&quality=low");
  await expect(page.getByTestId("clock")).toContainText("LIVE", { timeout: 60_000 });
  await page.getByTestId("scenario-picker").selectOption("txt2fly-demo");
  await expect(page.getByTestId("scenario-picker")).toHaveValue("txt2fly-demo", { timeout: 30_000 });
  await expect(page.getByTestId("clock")).toContainText("LIVE", { timeout: 60_000 });

  await page.getByTestId("plan-tab").click();
  // A circle the aircraft cannot hold, round a church inside a no-fly area: widened to a loop
  // round it, clear of the village (docs/06 §17), with every change explained.
  await page.getByTestId("typed-plan").click();
  await page.getByRole("button", { name: "Church, 30 m circle" }).click();
  await page.getByTestId("plan-check").click();
  await expect(page.getByTestId("plan-notes")).toContainText("the tightest it can", { timeout: 30_000 });
  await expect(page.getByTestId("plan-notes")).toContainText("keeps clear of Wrenfold village (no-fly)");

  const ask = async (text: string) => {
    await page.getByTestId("ask-input").fill(text);
    await page.getByTestId("ask-send").click();
  };
  await ask("turn off the geofence");
  await expect(page.getByTestId("conversation")).toContainText("can't be changed", { timeout: 20_000 });
  await ask("patrol fields A, B and C at 110 m");
  await expect(page.getByTestId("conversation")).toContainText("Plan: Patrol fields A, B and C", { timeout: 30_000 });
  await expect(page.getByTestId("plan-verdict")).toHaveText("Can fly", { timeout: 30_000 });
  await page.getByTestId("plan-confirm").click();
  await expect(page.getByTestId("active-plan")).toContainText("Patrol fields A, B and C", { timeout: 20_000 });
  await expect(page.getByTestId("ticker").getByText("Mode AUTO")).toBeVisible({ timeout: 20_000 });
  expect(errors).toEqual([]);
});
