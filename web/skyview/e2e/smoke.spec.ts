import { expect, test } from "@playwright/test";

test("plays a replay: the HUD airspeed changes and the console stays clean", async ({ page }) => {
  const errors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") errors.push(msg.text());
  });
  page.on("pageerror", (err) => errors.push(err.message));

  await page.goto("/?replay=fixtures/smoke&tiles=0");
  const ias = page.getByTestId("hud-IAS");
  await expect(ias).toBeVisible({ timeout: 30_000 });
  const clock = page.getByTestId("clock");
  const before = { ias: await ias.textContent(), clock: await clock.textContent() };
  await page.waitForTimeout(5_000); // playback starts automatically
  const after = { ias: await ias.textContent(), clock: await clock.textContent() };

  expect(after.clock).not.toBe(before.clock);
  expect(after.ias).not.toBe(before.ias);
  await expect(page.getByText("Fault: motor cut")).toHaveCount(0); // fixture has no faults
  expect(errors).toEqual([]);
});
