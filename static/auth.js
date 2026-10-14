/* If the server needs an access key (RETAILVISION_API_KEY), a 401 makes every fetch on this page
   ask for it once, exchange it for an HttpOnly cookie via /login, and retry the request. */
(function () {
    const realFetch = window.fetch.bind(window);
    let loginInFlight = null;

    function askAndLogin() {
        if (!loginInFlight) {
            loginInFlight = (async () => {
                const key = window.prompt("This RetailVision server needs an access key:");
                if (!key) return false;
                const res = await realFetch("/login", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ key }),
                });
                return res.ok;
            })().finally(() => { setTimeout(() => { loginInFlight = null; }, 0); });
        }
        return loginInFlight;
    }

    window.fetch = async function (input, init) {
        const res = await realFetch(input, init);
        const url = typeof input === "string" ? input : (input && input.url) || "";
        const sameOrigin = url.startsWith("/") || url.startsWith(window.location.origin);
        if (res.status === 401 && sameOrigin && !url.startsWith("/login")) {
            if (await askAndLogin()) return realFetch(input, init);
        }
        return res;
    };
})();
