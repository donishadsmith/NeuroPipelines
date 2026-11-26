// @ts-check

const FORM = /** @type {HTMLElement} */ (document.getElementById("lookup_form"));
const SPINNER = /** @type {HTMLElement} */ (document.getElementById("spinner"));
const STATUS_TEXT = /** @type {HTMLElement} */ (document.getElementById("status"));
const TEXT_OUTPUT = /** @type {HTMLElement} */ (document.getElementById("text_output"));
const FIGURES = /** @type {HTMLElement} */ (document.getElementById("figures"));
const TRANSFORM_TEXT = /** @type {HTMLElement} */ (document.getElementById("transform_text"));
const ERROR = /** @type {HTMLElement} */ (document.getElementById("error"));
const ERROR_MESSAGE = /** @type {HTMLElement} */ (document.getElementById("error_message"));

/** @type {boolean} */
let SUBMISSION_IN_PROGRESS = false;

class PostRequest {
    /**
     * @param {string} cohort
     * @param {number} x
     * @param {number} y
     * @param {number} z
     * @param {string} coordinate_space
     * @param {string} transform_method
     */
    constructor(cohort, x, y, z, coordinate_space, transform_method) {
        this.cohort = cohort;
        this.x = x;
        this.y = y;
        this.z = z;
        this.coordinate_space = coordinate_space;
        this.transform_method = transform_method;
    }

    display() {
        console.log(`Cohort: ${this.cohort}; X: ${this.x}; Y: ${this.y} Z: ${this.z}; Coordinate Space: ${this.coordinate_space}; Transform Method: ${this.transform_method}`);
    }

    /**
     * @returns {Object}
     */
    to_hashmap() {
        return {
            cohort: this.cohort,
            x: this.x,
            y: this.y,
            z: this.z,
            coordinate_space: this.coordinate_space,
            transform_method: this.transform_method
        }
    }
}

/**
 * @typedef {Object} RequestResult
 * @property {Response | null} response
 * @property {string | null} error_message
 */

/**
 * @param {PostRequest} post_request
 * @returns {Promise<RequestResult>}
 */
async function handle_request(post_request) {
    let response = null;
    let error_message = null;
    try {
        response = await fetch("/lookup", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(post_request.to_hashmap())
        });

        if (!response.ok) {
            const detail = (await response.json()).detail;

            if (typeof detail === "string") {
                error_message = detail;
            } else if (Array.isArray(detail)) {
                error_message = detail.map((item) => item.msg).join("; ");
            } else {
                error_message = `Request failed with status ${response.status}.`;
            }
        }
    } catch(error) {
        error_message = error instanceof Error ? error.message : String(error);
    }

    return {response: response, error_message: error_message};
}

function reset_status() {
    SPINNER.hidden = true;
    STATUS_TEXT.textContent = "";
}

function reset_output() {
    TRANSFORM_TEXT.textContent = "";
    TEXT_OUTPUT.textContent = "";
    FIGURES.replaceChildren();
}

/**
 * @param {Response | null} response
 * @param {string | null} error_message
 */
async function render_content(response, error_message) {
    reset_output();

    try {
        if (error_message || !response) {
            ERROR.hidden = false;
            ERROR_MESSAGE.textContent = error_message;

            return;
        }
        
        let body = (await response.json());
        TRANSFORM_TEXT.textContent = body.coordinate_transform_text !== "Placeholder" ? body.coordinate_transform_text : "";
        TEXT_OUTPUT.textContent = body.afni_text;

        for (const figure of body.atlases) {
            if (!figure) {
                continue;
            }

            FIGURES.insertAdjacentHTML("beforeend", 
            `
                <figure>
                    <img src="data:image/png;base64,${figure}">
                </figure>
            `
            );
        }

        if (body.atlases.some(Boolean)) {
            FIGURES.insertAdjacentHTML("beforeend", 
            `
                <p>Note: In each figure, the radius of the red sphere matches the distance from the coordinate to the closest label in that atlas.</p>
            `
            );
        }

    } catch (error) {
        console.error(error);
        ERROR.hidden = false;
        ERROR_MESSAGE.textContent = "Something went wrong showing the result.";
    } finally {
        reset_status();
        SUBMISSION_IN_PROGRESS = false;
    }
}

/**
 * @param {SubmitEvent} press
 */
async function handle_submission(press) {
    press.preventDefault();

    if (SUBMISSION_IN_PROGRESS) {
        return;
    }
    
    ERROR_MESSAGE.textContent = "";
    ERROR.hidden = true;
    SPINNER.hidden = false;
    STATUS_TEXT.textContent = "Running...";
    SUBMISSION_IN_PROGRESS = true;

    const post_request = new PostRequest(
        /** @type {HTMLSelectElement} */ (document.getElementById("cohort")).value,
        /** @type {HTMLInputElement} */ (document.getElementById("x")).valueAsNumber,
        /** @type {HTMLInputElement} */ (document.getElementById("y")).valueAsNumber,
        /** @type {HTMLInputElement} */ (document.getElementById("z")).valueAsNumber,
        /** @type {HTMLSelectElement} */ (document.getElementById("coordinate_space")).value,
        /** @type {HTMLSelectElement} */ (document.getElementById("transform_method")).value,
    );

    post_request.display();

    const { response, error_message } = await handle_request(post_request);

    render_content(response, error_message);
}

(() => {
    FORM.addEventListener("submit", handle_submission);
})();
