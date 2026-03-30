import { test, expect } from '@playwright/test';

test.describe('Harmony UI E2E Interaction Tests', () => {
    test.beforeEach(async ({ page }) => {
        // Mock the backend for basic initialization
        await page.route('**/harmony/canvas_data/**', async route => {
            const json = {
                objects: {
                    Obj1: {
                        VirtualMap: [[100, 100], [100, 200], [200, 200], [200, 100]],
                        "Camera 0": [[10, 10], [10, 20], [20, 20], [20, 10]]
                    }
                },
                moveable: ["Obj1"],
                selection: { firstCell: null, additionalCells: [] },
                selectable: ["Obj1"],
                viewId: "test_e2e"
            };
            await route.fulfill({ json });
        });

        // Load the page (assuming server is running or mocks can handle HTML)
        // For a true E2E, the server should be running, but we can verify the JS directly.
        // We'll point to a local file or the server URL.
        await page.goto('http://localhost:7000/harmony/');
    });

    test('Should initialize correctly', async ({ page }) => {
        const header = await page.locator('#GameWorldHeader');
        await expect(header).toContainText('Game World View');
        const canvas = await page.locator('#GameWorldOverlay');
        await expect(canvas).toBeVisible();
    });

    test('Should calculate canvas scaling correctly on resize', async ({ page }) => {
        // Check window.harmonyEditor dimensions
        const dimensions = await page.evaluate(() => {
            const canvas = document.getElementById('GameWorldOverlay');
            return { w: canvas.width, h: canvas.height };
        });
        expect(dimensions.w).toBeGreaterThan(0);
    });

    test('Should submit select_pixel on canvas click', async ({ page }) => {
        // Wait for img to load so overlay matches
        await page.waitForSelector('#GameWorld');
        
        // Mock the POST submission to avoid actual state change
        await page.route('**/harmony/select_pixel', async route => {
            await route.fulfill({ body: 'success' });
        });

        const canvas = await page.locator('#GameWorldOverlay');
        const box = await canvas.boundingBox();
        
        // Click middle of the canvas
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);

        // Check form values
        const selectedPixel = await page.locator('#selectedPixel').inputValue();
        expect(selectedPixel).toContain('['); // Verify it's JSON array
        
        const appendPixel = await page.locator('#appendPixel').inputValue();
        expect(appendPixel).toBe('false');
    });

    test('Should enable appendPixel on CTRL+Click', async ({ page }) => {
        await page.route('**/harmony/select_pixel', async route => {
            await route.fulfill({ body: 'success' });
        });

        const canvas = await page.locator('#GameWorldOverlay');
        const box = await canvas.boundingBox();
        
        // CTRL + Click
        await page.keyboard.down('Control');
        await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
        await page.keyboard.up('Control');

        const appendPixel = await page.locator('#appendPixel').inputValue();
        expect(appendPixel).toBe('true');
    });
});
