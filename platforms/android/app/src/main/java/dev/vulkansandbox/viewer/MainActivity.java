package dev.vulkansandbox.viewer;

import android.app.AlertDialog;
import com.google.androidgamesdk.GameActivity;

/** Android owns the Activity/window; the shared renderer owns Vulkan resources. */
public final class MainActivity extends GameActivity {
    static { System.loadLibrary("sandbox_android"); }

    // Called on the native thread. Never show Android UI from that thread directly.
    public void showRendererError(final String message) {
        runOnUiThread(() -> {
            if (isFinishing() || isDestroyed()) return;
            new AlertDialog.Builder(this)
                .setTitle("Unable to start Vulkan Sandbox")
                .setMessage(message)
                .setPositiveButton("Close", (dialog, which) -> finish())
                .setCancelable(false)
                .show();
        });
    }
}
