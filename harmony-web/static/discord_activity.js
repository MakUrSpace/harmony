import { DiscordSDK } from "https://esm.sh/@discord/embedded-app-sdk@1.2.0";

let discordSdk;

async function setupDiscordSdk() {
    const clientIdMeta = document.querySelector('meta[name="discord_client_id"]');
    if (!clientIdMeta) {
        console.error("No discord_client_id meta tag found.");
        return;
    }
    const clientId = clientIdMeta.content;

    if (!clientId) {
        console.warn("Discord Client ID is not configured. Activity initialization skipped.");
        return;
    }

    discordSdk = new DiscordSDK(clientId);

    await discordSdk.ready();
    console.log("Discord SDK is ready");

    const { code } = await discordSdk.commands.authorize({
        client_id: clientId,
        response_type: "code",
        state: "",
        prompt: "none",
        scope: ["identify", "guilds"],
    });

    const response = await fetch("/api/discord/token-exchange", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            code,
        }),
    });

    if (!response.ok) {
        throw new Error("Failed to exchange code for token");
    }

    const { access_token } = await response.json();

    const auth = await discordSdk.commands.authenticate({
        access_token,
    });

    if (!auth) {
        throw new Error("Authenticate command failed");
    }

    console.log("Successfully authenticated!", auth);
}

document.addEventListener('DOMContentLoaded', () => {
    setupDiscordSdk().catch(console.error);
});
