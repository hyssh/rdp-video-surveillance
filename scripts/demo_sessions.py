"""Synthetic RDP session stories for the demo (all names, hosts and data are fictional)."""
from typing import Any, Dict, List, Optional, Tuple

DATE = "10/14/2025"


def win(title: str, body: List[Tuple], theme: str = "light", toolbar: str = "", rect: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
    return {"title": title, "body": body, "theme": theme, "toolbar": toolbar, "rect": rect}


def term(title: str, lines: List[Any]) -> Dict[str, Any]:
    return win(title, [("code",) + (l if isinstance(l, tuple) else (l,)) for l in lines], theme="dark")


def listing(title: str, path: str, rows: List[Tuple], sel: Optional[int] = None, hdr: Tuple = ("Name", "Size", "Date modified")) -> Dict[str, Any]:
    widths = [52, 18, 30]
    body = [("hrow", list(hdr), widths)] + [("row", list(r), widths, i == sel) for i, r in enumerate(rows)]
    if not rows:
        body.append(("t", "This folder is empty.", "gray"))
    return win(title, body, toolbar=path)


def creds(host: str, user: str, pw_len: int = 12, title: str = "Windows Security") -> Dict[str, Any]:
    return win(title, [
        ("h", "Enter your credentials"),
        ("t", f"These credentials will be used to connect to {host}."),
        ("sp", 6),
        ("field", "User name", user),
        ("field", "Password", "\u2022" * pw_len),
        ("btns", ["OK", "Cancel"]),
    ], rect=(330, 150, 950, 470))


def frame(w: Dict[str, Any], desc: str, risk: int, action: str, dialog: Optional[Dict[str, Any]] = None, toast: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
    return {"win": w, "desc": desc, "risk": risk, "action": action, "dialog": dialog, "toast": toast}


def dlg(title: str, body: List[Tuple], size: Tuple[int, int] = (560, 250)) -> Dict[str, Any]:
    return {"title": title, "body": body, "size": size}


def sql_win(title: str, query: str, grid: Optional[List[Tuple]] = None, status: str = "") -> Dict[str, Any]:
    body: List[Tuple] = [("code", query), ("sp", 10)]
    body += grid or []
    if status:
        body += [("sp", 6), ("t", status, "gray")]
    return win(title, body)


def ended(message: str) -> Dict[str, Any]:
    return win("Remote Desktop Connection", [("h", "Session ended"), ("t", message), ("sp", 10), ("btns", ["OK"])], rect=(330, 190, 950, 430))


# ---------------------------------------------------------------- A: high / critical risk
A_GRID_HDR = ("hrow", ["customer_id", "full_name", "email", "phone", "card_last4"], [14, 22, 30, 20, 14])
A_ROWS = [("100231", "Alicia Moreno", "a.moreno@example.com", "+1 425-555-0142", "4417"),
          ("100232", "Daniel Okafor", "d.okafor@example.com", "+1 206-555-0187", "9082"),
          ("100233", "Priya Raman", "priya.r@example.com", "+1 253-555-0119", "3350"),
          ("100234", "Lukas Brandt", "l.brandt@example.com", "+1 360-555-0175", "7721")]
A_GRID = [A_GRID_HDR] + [("row", list(r), [14, 22, 30, 20, 14]) for r in A_ROWS]
DBTITLE = "DB Admin Console - PROD-DB-01 (sa_vendor)"
PS_ADMIN = "Administrator: Windows PowerShell"

SESSION_A = {
    "slug": "vendor_db_admin_session",
    "filename": "vendor_db_admin_session.mp4",
    "host": "PROD-DB-01", "user": "CONTOSO\\vendor.mlopez", "start": (22, 47),
    "security_level": "critical", "risk_score": 94, "age_hours": 2,
    "summary": ("A vendor engineer (CONTOSO\\vendor.mlopez) connected to PROD-DB-01 at 10:47 PM under change ticket CHG-48213, "
                "which only covers index rebuilds on CustomerDB (frame 2 @ 00:00:06). After a legitimate ALTER INDEX run (frame 5 @ 00:00:15) the user queried "
                "customer PII, exported the full dbo.Customers table (184,203 rows) to C:\\Temp\\customers_export_full.csv (frames 9-10), encrypted it into a password-protected zip (frame 12), "
                "copied it to a USB drive BACKUP_USB (E:) (frames 13-15) and uploaded it to the external file-sharing site files.shared-drop.example, producing a public share link (frames 16-18). "
                "The user then cleared the Windows Security event log, deleted the exported files and wiped PowerShell history (frames 19-22) before signing out."),
    "manual_review": "Yes",
    "manual_review_reason": "Bulk export of customer PII outside the approved change scope, copied to removable media and an external site, followed by log clearing and cleanup.",
    "security_risk_score": "Critical: a full customer table was exfiltrated two ways (USB and external upload) and audit evidence was destroyed afterwards.",
    "key_frames": [
        (2, "Approved change ticket scope is index maintenance only; no data export is authorised.", "ITSM ticket CHG-48213 shown in Edge.", "Scope: Rebuild fragmented indexes on CustomerDB (no data export)", 0.95),
        (3, "Logged in to the database admin console with the shared SQL account sa_vendor.", "Connect to Server dialog, SQL Server Authentication.", "Login: sa_vendor, password field masked", 0.92),
        (6, "Customer PII (names, emails, phones, card last four digits) displayed in query results.", "SELECT TOP 10 * FROM dbo.Customers in CustomerDB.", "Result grid columns customer_id, full_name, email, phone, card_last4", 0.95),
        (8, "Full-table query against the customer table, outside the index maintenance scope.", "SELECT * FROM dbo.Customers running (184,203 rows).", "Status bar: Executing query... 184,203 rows", 0.9),
        (10, "Customer table exported to a CSV file on the server.", "Export Data Wizard completed.", "Export successful - 184,203 rows exported to C:\\Temp\\customers_export_full.csv", 0.95),
        (12, "Export file compressed into a password-protected AES-256 zip archive.", "7-Zip Add to Archive dialog.", "Archive: customers_export_full.zip, Encryption method AES-256", 0.93),
        (14, "Archive copied to a removable USB drive.", "File copy dialog to BACKUP_USB (E:).", "Copying 1 item from Temp to BACKUP_USB (E:) - 62% complete", 0.95),
        (17, "Archive uploaded to an external file-sharing website.", "SharedDrop upload page in Edge.", "customers_export_full.zip 12.3 MB - 74% uploaded", 0.95),
        (18, "Public share link generated for the uploaded archive.", "SharedDrop upload complete page.", "Share link: https://files.shared-drop.example/s/9fK2xQ", 0.97),
        (19, "Security and Application event logs cleared via wevtutil.", "Administrator PowerShell.", "wevtutil cl Security", 0.95),
        (20, "Event Viewer shows the Security log was cleared (event 1102) by the same account.", "Event Viewer, Security log.", "Event 1102: The audit log was cleared. Account Name: vendor.mlopez", 0.96),
        (21, "Exported files deleted and PowerShell history removed.", "Administrator PowerShell.", "Remove-Item C:\\Temp\\customers_export_full.* -Force; Clear-History", 0.94),
    ],
    "frames": [
        frame(creds("PROD-DB-01.contoso.local", "CONTOSO\\vendor.mlopez"),
              "Windows Security credentials prompt for Remote Desktop to PROD-DB-01.contoso.local. User name CONTOSO\\vendor.mlopez is entered and the password field is masked; OK and Cancel buttons are visible. The session starts at 10:47 PM.", 30, "login"),
        frame(win("Server Manager - PROD-DB-01", [("h", "WELCOME TO SERVER MANAGER"), ("t", "Dashboard  |  Local Server  |  All Servers  |  File and Storage Services"), ("sp", 10),
                                                   ("b", "ROLES AND SERVER GROUPS"), ("t", "Roles: 2 | Server groups: 1 | Servers total: 1"), ("t", "Local Server - Manageability: Online"), ("t", "Performance alerts: none")]),
              "Server Manager dashboard on PROD-DB-01 after sign-in. Roles: 2, Servers total: 1, manageability Online. No user activity beyond landing on the desktop.", 8, "other"),
        frame(win("CHG-48213 - Quarterly index maintenance - Contoso ITSM - Microsoft Edge", [
                  ("h", "CHG-48213: Quarterly index maintenance"), ("field", "Status", "Approved"), ("field", "Window", "10/14/2025 22:30 - 23:30"),
                  ("field", "Requester", "Contoso Managed Services (vendor)"), ("t", "Scope: Rebuild fragmented indexes on CustomerDB (no data export)."), ("t", "Systems: PROD-DB-01"), ("t", "Approved by: R. Patel, Change Advisory Board")],
                  toolbar="https://itsm.contoso.local/change/CHG-48213"),
              "Microsoft Edge shows ITSM change ticket CHG-48213 'Quarterly index maintenance', status Approved, window 22:30-23:30 on PROD-DB-01. The scope states: rebuild fragmented indexes on CustomerDB, no data export.", 12, "browsing"),
        frame(win("DB Admin Console - Connect to Server", [("h", "Connect to Server"), ("field", "Server name", "PROD-DB-01"), ("field", "Authentication", "SQL Server Authentication"),
                                                           ("field", "Login", "sa_vendor"), ("field", "Password", "\u2022" * 14), ("btns", ["Connect", "Cancel"])], rect=(260, 100, 1020, 560)),
              "DB Admin Console 'Connect to Server' dialog: server PROD-DB-01, SQL Server Authentication, login sa_vendor with a masked password. The user connects with a shared privileged SQL account.", 42, "login"),
        frame(win(DBTITLE, [("b", "Object Explorer"), ("t", "PROD-DB-01 (SQL Server 15.0.4345)"), ("t", "    Databases"), ("t", "        CustomerDB"), ("t", "        BillingDB"),
                            ("t", "        master"), ("t", "        msdb"), ("t", "    Security / Server Objects / SQL Server Agent")]),
              "DB Admin Console connected to PROD-DB-01 as sa_vendor. Object Explorer lists databases CustomerDB, BillingDB, master and msdb.", 38, "admin_console"),
        frame(sql_win("SQLQuery1.sql - CustomerDB - " + DBTITLE, "ALTER INDEX ALL ON dbo.Customers REBUILD;", [("t", "Messages", "gray"), ("t", "Commands completed successfully."), ("t", "Total execution time: 00:00:04.312")]),
              "Query window on CustomerDB runs 'ALTER INDEX ALL ON dbo.Customers REBUILD;' and returns 'Commands completed successfully' in 4.3 seconds. This matches the approved index maintenance scope.", 20, "admin_console"),
        frame(sql_win("SQLQuery2.sql - CustomerDB - " + DBTITLE, "SELECT TOP 10 * FROM dbo.Customers;", A_GRID, "4 of 10 rows shown"),
              "Query 'SELECT TOP 10 * FROM dbo.Customers;' returns customer PII in the result grid: customer_id, full_name (e.g. Alicia Moreno), email, phone and card_last4. Sensitive customer data is visible on screen.", 55, "admin_console"),
        frame(sql_win("SQLQuery3.sql - CustomerDB - " + DBTITLE, "SELECT COUNT(*) FROM dbo.Customers;", [("hrow", ["(No column name)"], [100]), ("row", ["184203"], [100])], "1 row(s) returned"),
              "Query 'SELECT COUNT(*) FROM dbo.Customers;' returns 184203, showing the user is sizing the customer table before a bulk read.", 48, "admin_console"),
        frame(sql_win("SQLQuery4.sql - CustomerDB - " + DBTITLE, "SELECT * FROM dbo.Customers;", [("progress", "Executing query... 184,203 rows", 71)], "Elapsed 00:00:07"),
              "Query 'SELECT * FROM dbo.Customers;' is executing against the full customer table with an elapsed time of 7 seconds and 184,203 rows being returned, 71% complete.", 62, "admin_console"),
        frame(win(DBTITLE, [("t", "SQLQuery4.sql - results ready", "gray")]),
              "Export Data Wizard opened from the DB console: source CustomerDB.dbo.Customers, destination Flat File, file C:\\Temp\\customers_export_full.csv. Back, Next and Finish buttons are shown.", 72, "file_transfer",
              dialog=dlg("Export Data Wizard", [("h", "Choose a destination"), ("field", "Source", "CustomerDB.dbo.Customers"), ("field", "Destination", "Flat File Destination"),
                                                ("field", "File name", r"C:\Temp\customers_export_full.csv"), ("btns", ["Finish", "Next", "Back"])], (700, 300))),
        frame(win(DBTITLE, [("t", "SQLQuery4.sql - results ready", "gray")]),
              "Export Data Wizard reports 'Export successful - 184,203 rows exported' to C:\\Temp\\customers_export_full.csv. The complete customer table now exists as a plain CSV file on the server.", 72, "file_transfer",
              dialog=dlg("Export Data Wizard - Complete", [("h", "Export successful"), ("t", "184,203 rows exported to C:\\Temp\\customers_export_full.csv"), ("progress", "Operation", 100), ("btns", ["Close"])], (640, 260))),
        frame(listing("Temp - File Explorer", r"C:\Temp", [("customers_export_full.csv", "41,812 KB", "10/14/2025 10:58 PM"), ("setup.log", "3 KB", "10/14/2025 10:31 PM")], 0),
              "File Explorer at C:\\Temp shows customers_export_full.csv (41,812 KB, modified 10:58 PM) selected next to setup.log.", 68, "file_transfer"),
        frame(win("7-Zip - Add to Archive", [("field", "Archive", "customers_export_full.zip"), ("field", "Archive format", "zip"), ("field", "Compression level", "Normal"),
                                              ("field", "Encryption method", "AES-256"), ("field", "Enter password", "\u2022" * 12), ("btns", ["OK", "Cancel"])], rect=(260, 90, 1020, 560)),
              "7-Zip 'Add to Archive' dialog creating customers_export_full.zip with AES-256 encryption and a masked password. The exported CSV is being packaged into an encrypted archive.", 80, "file_transfer"),
        frame(listing("Temp - File Explorer", r"C:\Temp", [("customers_export_full.csv", "41,812 KB", "10/14/2025 10:58 PM"), ("customers_export_full.zip", "12,304 KB", "10/14/2025 11:03 PM"), ("setup.log", "3 KB", "10/14/2025 10:31 PM")], 1),
              "C:\\Temp now contains customers_export_full.zip (12,304 KB). A notification reports 'BACKUP_USB (E:) - Device connected', so a removable drive was just attached.", 78, "file_transfer",
              toast=("BACKUP_USB (E:)", "Device connected - tap to choose what happens")),
        frame(win("Temp - File Explorer", [("t", "C:\\Temp", "gray")]),
              "A copy dialog shows 'Copying 1 item from Temp to BACKUP_USB (E:)': customers_export_full.zip, 62% complete, 7.6 MB of 12.3 MB. The encrypted archive is being written to the USB drive.", 88, "file_transfer",
              dialog=dlg("Copying 1 item from Temp to BACKUP_USB (E:)", [("t", "customers_export_full.zip"), ("progress", "62% complete", 62), ("t", "Items remaining: 1 (4.7 MB)"), ("btns", ["Cancel"])], (600, 240))),
        frame(listing("BACKUP_USB (E:) - File Explorer", r"E:\\", [("customers_export_full.zip", "12,304 KB", "10/14/2025 11:03 PM")], 0),
              "File Explorer on BACKUP_USB (E:) shows customers_export_full.zip (12,304 KB), confirming the archive now exists on the removable drive.", 85, "file_transfer"),
        frame(win("SharedDrop - Secure file sharing - Microsoft Edge", [("h", "SharedDrop"), ("t", "Share files up to 2 GB. No account required. Links expire in 7 days."), ("sp", 10),
                                                                         ("t", "Drop files here or click to browse"), ("btns", ["Choose files", "Upload"], "left")], toolbar="https://files.shared-drop.example/upload"),
              "Microsoft Edge on SharedDrop (files.shared-drop.example/upload), an anonymous external file-sharing site: 'No account required. Links expire in 7 days.' with Choose files and Upload buttons.", 80, "file_transfer"),
        frame(win("SharedDrop - Uploading - Microsoft Edge", [("h", "Uploading"), ("t", "customers_export_full.zip  (12.3 MB)"), ("progress", "Upload progress", 74), ("t", "Keep this tab open until the upload completes.")],
                  toolbar="https://files.shared-drop.example/upload"),
              "SharedDrop shows customers_export_full.zip (12.3 MB) uploading, 74% complete. The encrypted customer archive is being sent to an external site.", 92, "file_transfer"),
        frame(win("SharedDrop - Upload complete - Microsoft Edge", [("h", "Upload complete"), ("t", "Your file is ready to share."), ("field", "Share link", "https://files.shared-drop.example/s/9fK2xQ"), ("btns", ["Copy link", "Done"], "left")],
                  toolbar="https://files.shared-drop.example/done"),
              "SharedDrop 'Upload complete' page displays the public share link https://files.shared-drop.example/s/9fK2xQ with a Copy link button. Anyone with the link can download the customer archive.", 95, "file_transfer"),
        frame(term(PS_ADMIN, ["PS C:\\Windows\\system32> wevtutil cl Security", "PS C:\\Windows\\system32> wevtutil cl Application", "PS C:\\Windows\\system32> wevtutil el | Measure-Object", "Count    : 412",
                              ("PS C:\\Windows\\system32> _", "green")]),
              "Administrator PowerShell runs 'wevtutil cl Security' and 'wevtutil cl Application', clearing the Windows Security and Application event logs without errors.", 93, "terminal"),
        frame(win("Event Viewer - Security", [("hrow", ["Level", "Date and Time", "Source", "Event ID"], [18, 32, 30, 20]), ("row", ["Information", "10/14/2025 11:15:42 PM", "Eventlog", "1102"], [18, 32, 30, 20], True),
                                              ("sp", 8), ("b", "Event 1102, Eventlog"), ("t", "The audit log was cleared."), ("t", "Subject: Security ID CONTOSO\\vendor.mlopez, Account Name vendor.mlopez")]),
              "Event Viewer Security log contains a single entry, Event ID 1102 'The audit log was cleared', Subject account vendor.mlopez, at 11:15:42 PM. All previous security events are gone.", 94, "config_change"),
        frame(term(PS_ADMIN, ["PS C:\\Windows\\system32> Remove-Item C:\\Temp\\customers_export_full.* -Force", "PS C:\\Windows\\system32> Remove-Item (Get-PSReadlineOption).HistorySavePath",
                              "PS C:\\Windows\\system32> Clear-History", ("PS C:\\Windows\\system32> _", "green")]),
              "PowerShell deletes C:\\Temp\\customers_export_full.* with -Force, removes the PSReadline history file and runs Clear-History, erasing traces of the export on the server.", 90, "terminal"),
        frame(listing("Temp - File Explorer", r"C:\Temp", [("setup.log", "3 KB", "10/14/2025 10:31 PM")], None),
              "File Explorer at C:\\Temp now shows only setup.log; customers_export_full.csv and customers_export_full.zip have been removed from the server.", 55, "other"),
        frame(win("Temp - File Explorer", [("t", "C:\\Temp", "gray")]),
              "A 'Safe to Remove Hardware' notification for BACKUP_USB (E:) appears, so the USB drive carrying the archive is being ejected.", 75, "file_transfer",
              toast=("Safely Remove Hardware", "The device BACKUP_USB (E:) can now be safely removed from the computer.")),
        frame(win("Windows", [("t", "Server Manager - PROD-DB-01", "gray")]), "A 'Sign out' confirmation prompt is displayed for CONTOSO\\vendor.mlopez; the session is ending at about 11:19 PM.", 25, "other",
              dialog=dlg("Sign out", [("h", "Sign out of PROD-DB-01?"), ("t", "Your Remote Desktop session will end."), ("btns", ["Sign out", "Cancel"])], (480, 220))),
        frame(ended("Your Remote Desktop Services session has ended. Reason: You have been logged off by the user."),
              "Remote Desktop Connection shows 'Session ended: logged off by the user'. The recording ends with no further activity.", 5, "idle"),
    ],
}

# ---------------------------------------------------------------- B: low risk, routine patching
SESSION_B = {
    "slug": "vendor_routine_patching",
    "filename": "vendor_routine_patching.mp4",
    "host": "APP-WEB-03", "user": "CONTOSO\\svc.patch.vendor", "start": (10, 2),
    "security_level": "low", "risk_score": 12, "age_hours": 26,
    "summary": ("A vendor service account (CONTOSO\\svc.patch.vendor) connected to APP-WEB-03 during the approved window of change ticket CHG-48102 (monthly Windows patching, frame 2 @ 00:00:06). "
                "Three updates (KB5044273, KB5043935, KB5042101) were installed through Windows Update (frames 3-6), the server restarted and the user reconnected (frames 7-9). "
                "The user verified patch levels with Get-HotFix, restarted the W3SVC service, confirmed the /health endpoint and IIS site status, reviewed the event log and a patch log, "
                "updated and resolved the ticket (frames 10-18) and signed out. All activity matches the ticket scope; no sensitive data or unapproved tools were observed."),
    "manual_review": "No",
    "manual_review_reason": "Activity matches the approved change ticket; no sensitive data, data movement or unapproved software observed.",
    "security_risk_score": "Low: routine, ticketed patching with standard Windows tools and expected service restarts.",
    "key_frames": [
        (0, "Remote Desktop sign-in by the vendor patching service account.", "Windows Security credentials prompt for APP-WEB-03.", "User name: CONTOSO\\svc.patch.vendor, password masked", 0.9),
        (12, "Restarted the IIS World Wide Web Publishing Service as part of patch validation.", "PowerShell session on APP-WEB-03.", "Restart-Service W3SVC", 0.93),
    ],
    "frames": [
        frame(creds("APP-WEB-03.contoso.local", "CONTOSO\\svc.patch.vendor", 14),
              "Windows Security credentials prompt to connect to APP-WEB-03.contoso.local with user CONTOSO\\svc.patch.vendor; password masked. Session start at 10:02 AM.", 24, "login"),
        frame(win("Server Manager - APP-WEB-03", [("h", "WELCOME TO SERVER MANAGER"), ("t", "Roles: Web Server (IIS)  |  Servers total: 1"), ("t", "Local Server - Manageability: Online"), ("t", "Last installed updates: 09/09/2025")]),
              "Server Manager dashboard for APP-WEB-03 after sign-in: role Web Server (IIS), manageability Online, last installed updates 09/09/2025.", 5, "other"),
        frame(win("CHG-48102 - Monthly Windows patching - Contoso ITSM - Microsoft Edge", [("h", "CHG-48102: Monthly Windows patching"), ("field", "Status", "Approved"), ("field", "Window", "10/14/2025 10:00 - 12:00"),
                                                                                           ("t", "Scope: Install approved Windows updates and restart IIS services on APP-WEB-03."), ("t", "Rollback: restore previous VM snapshot."), ("t", "Approved by: R. Patel, Change Advisory Board")],
                  toolbar="https://itsm.contoso.local/change/CHG-48102"),
              "Edge shows ITSM change ticket CHG-48102 'Monthly Windows patching', Approved, window 10:00-12:00, scope: install approved Windows updates and restart IIS services on APP-WEB-03.", 8, "browsing"),
        frame(win("Settings - Windows Update", [("h", "Windows Update"), ("t", "Updates available", "gray"), ("t", "2025-10 Cumulative Update for Windows Server 2022 (KB5044273)"), ("t", ".NET Framework 4.8.1 Security Update (KB5043935)"),
                                                ("t", "Servicing Stack Update (KB5042101)"), ("btns", ["Install all", "Pause updates"], "left")]),
              "Settings > Windows Update lists three approved updates: 2025-10 Cumulative Update KB5044273, .NET Framework 4.8.1 security update KB5043935 and Servicing Stack Update KB5042101. Install all is available.", 10, "config_change"),
        frame(win("Settings - Windows Update", [("h", "Windows Update"), ("progress", "Downloading updates", 45), ("t", "2025-10 Cumulative Update (KB5044273) - downloading")]),
              "Windows Update is downloading updates, 45% complete, currently KB5044273.", 6, "config_change"),
        frame(win("Settings - Windows Update", [("h", "Windows Update"), ("progress", "Installing updates", 80), ("t", ".NET Framework 4.8.1 Security Update (KB5043935) - installing")]),
              "Windows Update is installing updates, 80% complete, currently the .NET Framework 4.8.1 security update KB5043935.", 6, "config_change"),
        frame(win("Settings - Windows Update", [("h", "Restart required"), ("t", "Updates installed. Restart to finish installing 3 updates."), ("btns", ["Restart now", "Schedule the restart"], "left")]),
              "Windows Update reports 'Restart required' after installing 3 updates, with buttons Restart now and Schedule the restart.", 8, "config_change"),
        frame(win("Settings - Windows Update", [("t", "Restart required", "gray")]), "A confirmation dialog 'Windows will restart' is shown; the user chooses Restart now to finish patching.", 10, "config_change",
              dialog=dlg("Restart Windows", [("h", "Restart now?"), ("t", "Remote sessions will be disconnected."), ("btns", ["Restart now", "Cancel"])], (480, 220))),
        frame(ended("Reconnecting... Attempt 1 of 20. Your session will resume when the server is available."),
              "Remote Desktop Connection shows a reconnecting message, 'Attempt 1 of 20', while APP-WEB-03 restarts after patching.", 4, "idle"),
        frame(creds("APP-WEB-03.contoso.local", "CONTOSO\\svc.patch.vendor", 14),
              "Windows Security credentials prompt again after the restart; user CONTOSO\\svc.patch.vendor re-authenticates to APP-WEB-03.", 20, "login"),
        frame(term("Windows PowerShell", ["PS C:\\Users\\svc.patch.vendor> Get-HotFix | Sort InstalledOn -Desc | Select -First 3", "", "HotFixID    InstalledOn     InstalledBy", "--------    -----------     -----------",
                                          "KB5044273   10/14/2025      CONTOSO\\svc.patch.vendor", "KB5043935   10/14/2025      CONTOSO\\svc.patch.vendor", "KB5042101   10/14/2025      CONTOSO\\svc.patch.vendor"]),
              "PowerShell 'Get-HotFix | Sort InstalledOn -Desc | Select -First 3' lists KB5044273, KB5043935 and KB5042101 installed 10/14/2025, confirming the patches applied.", 14, "terminal"),
        frame(term("Windows PowerShell", ["PS C:\\Users\\svc.patch.vendor> Get-Service W3SVC, WAS | Format-Table Name, Status", "", "Name   Status", "----   ------", "W3SVC  Running", "WAS    Running"]),
              "PowerShell 'Get-Service W3SVC, WAS' shows both IIS services (W3SVC and WAS) with status Running after the reboot.", 12, "terminal"),
        frame(term("Windows PowerShell", ["PS C:\\Users\\svc.patch.vendor> Restart-Service W3SVC", "PS C:\\Users\\svc.patch.vendor> Get-Service W3SVC", "", "Status   Name    DisplayName", "Running  W3SVC   World Wide Web Publishing Service"]),
              "PowerShell runs 'Restart-Service W3SVC' and the World Wide Web Publishing Service returns to Running, a standard validation step in the patch plan.", 18, "terminal"),
        frame(win("app-web-03 health - Microsoft Edge", [("code", '{"status": "Healthy", "version": "2.14.1", "uptime": "00:04:12"}')], toolbar="http://app-web-03.contoso.local/health"),
              "Edge opens http://app-web-03.contoso.local/health and shows {\"status\": \"Healthy\", \"version\": \"2.14.1\", \"uptime\": \"00:04:12\"}, confirming the web app is healthy after patching.", 6, "browsing"),
        frame(win("Internet Information Services (IIS) Manager", [("b", "Sites"), ("hrow", ["Name", "Status", "Binding"], [34, 22, 44]), ("row", ["Default Web Site", "Started (http)", "*:80"], [34, 22, 44], True),
                                                                   ("row", ["ContosoApp", "Started (http)", "*:8080"], [34, 22, 44]), ("t", "Application pool: ContosoAppPool - Started")]),
              "IIS Manager lists sites Default Web Site and ContosoApp, both Started, and application pool ContosoAppPool Started.", 8, "admin_console"),
        frame(win("Event Viewer - System", [("hrow", ["Level", "Date and Time", "Source", "Event ID"], [16, 32, 34, 18]), ("row", ["Information", "10/14/2025 10:41 AM", "WindowsUpdateClient", "19"], [16, 32, 34, 18], True),
                                           ("t", "Installation Successful: Windows successfully installed the following update: 2025-10 Cumulative Update (KB5044273)")]),
              "Event Viewer System log shows WindowsUpdateClient event 19 'Installation Successful' for 2025-10 Cumulative Update KB5044273 at 10:41 AM.", 8, "admin_console"),
        frame(listing("PatchLogs - File Explorer", r"C:\PatchLogs", [("PatchRun_2025-10-14.log", "14 KB", "10/14/2025 10:46 AM"), ("PatchRun_2025-09-09.log", "13 KB", "09/09/2025 10:51 AM")], 0),
              "File Explorer at C:\\PatchLogs shows PatchRun_2025-10-14.log (14 KB, 10:46 AM) and last month's PatchRun_2025-09-09.log.", 8, "other"),
        frame(win("PatchRun_2025-10-14.log - Notepad", [("code", "10:02 Patch window start CHG-48102"), ("code", "10:19 Installed KB5044273, KB5043935, KB5042101"), ("code", "10:36 Reboot complete"), ("code", "10:44 W3SVC restarted, /health = Healthy")]),
              "Notepad shows PatchRun_2025-10-14.log with entries: patch window start CHG-48102, installed KB5044273/KB5043935/KB5042101, reboot complete, W3SVC restarted, /health = Healthy.", 8, "document_edit"),
        frame(win("CHG-48102 - Monthly Windows patching - Contoso ITSM - Microsoft Edge", [("h", "CHG-48102: Monthly Windows patching"), ("field", "Status", "Resolved"), ("t", "Comment: Patching completed. 3 KBs installed, server restarted, services and /health verified."),
                                                                                           ("btns", ["Add comment", "Close ticket"], "left")], toolbar="https://itsm.contoso.local/change/CHG-48102"),
              "The ITSM ticket CHG-48102 is updated with the comment 'Patching completed. 3 KBs installed, server restarted, services and /health verified.' and status Resolved.", 6, "browsing"),
        frame(win("CHG-48102 - Contoso ITSM - Microsoft Edge", [("t", "Ticket updated", "gray")]), "A 'Close ticket' confirmation is shown for CHG-48102 and the user confirms.", 5, "browsing",
              dialog=dlg("Close ticket", [("h", "Close CHG-48102?"), ("t", "The change will be marked Completed."), ("btns", ["Close ticket", "Cancel"])], (480, 220))),
        frame(win("Windows", [("t", "APP-WEB-03", "gray")]), "A 'Sign out' prompt for APP-WEB-03 is displayed; the session ends at about 11:08 AM.", 5, "other",
              dialog=dlg("Sign out", [("h", "Sign out of APP-WEB-03?"), ("btns", ["Sign out", "Cancel"])], (460, 200))),
        frame(ended("Your Remote Desktop Services session has ended. Reason: You have been logged off by the user."), "Remote Desktop Connection shows 'Session ended: logged off by the user'.", 3, "idle"),
    ],
}

# ---------------------------------------------------------------- C: medium risk, mixed session
SESSION_C = {
    "slug": "contractor_mixed_session",
    "filename": "contractor_mixed_session.mp4",
    "host": "WS-FIN-12", "user": "CONTOSO\\ctr.pnguyen", "start": (14, 15),
    "security_level": "medium", "risk_score": 46, "age_hours": 50,
    "summary": ("A finance contractor (CONTOSO\\ctr.pnguyen) worked mostly on expected tasks: Outlook, the Q3_Forecast_v4.xlsx workbook, Jira ticket FIN-2231 and a Confluence checklist (frames 1-6). "
                "The user then downloaded an unrecognised tool, invoicegen_setup.exe, which Windows Security quarantined (frames 7-9); real-time protection was turned off (frames 10-11), the installer was run with an unknown publisher (frames 12-14) and protection was turned back on about 12 seconds later (frame 15). "
                "Later the user browsed \\\\FILESRV01\\HR$ and opened Salary_Bands_2025.xlsx, which is unrelated to the finance task (frames 17-19), before returning to Finance$ and emailing the forecast (frames 20-21)."),
    "manual_review": "Yes",
    "manual_review_reason": "Antivirus real-time protection was disabled to run an unsigned installer, and the user opened HR compensation files unrelated to the assigned work.",
    "security_risk_score": "Medium: mostly benign work, with two borderline actions (brief antivirus disablement and access to an unrelated HR share). No data movement was observed.",
    "key_frames": [
        (7, "Visited an unfamiliar download page for a third-party utility.", "Edge showing InvoiceGen Utility download page.", "InvoiceGen Utility - Free Download at invoicegen-tools.example", 0.88),
        (9, "Windows Security quarantined the downloaded installer as potentially unwanted software.", "Windows Security notification.", "Threat blocked: PUA:Win32/InvoiceGen - invoicegen_setup.exe quarantined", 0.93),
        (11, "Real-time protection was switched off in Windows Security.", "Virus & threat protection settings.", "Real-time protection toggle Off - 'Your device is vulnerable'", 0.95),
        (13, "Installer from an unknown publisher launched with an elevation prompt.", "User Account Control prompt.", "Verified publisher: Unknown - invoicegen_setup.exe", 0.9),
        (15, "Real-time protection was re-enabled.", "Virus & threat protection settings.", "Real-time protection toggle On - 'Your device is being protected'", 0.95),
        (17, "Browsed the HR network share, which is unrelated to the finance task.", "File Explorer on \\\\FILESRV01.", "Shares: Finance$, HR$, IT$, Public", 0.85),
        (19, "Opened an HR compensation workbook.", "Excel with Salary_Bands_2025.xlsx.", "Salary_Bands_2025.xlsx - Level, Min, Mid, Max", 0.92),
    ],
    "frames": [
        frame(creds("WS-FIN-12.contoso.local", "CONTOSO\\ctr.pnguyen", 11),
              "Windows Security credentials prompt for WS-FIN-12.contoso.local with user CONTOSO\\ctr.pnguyen and a masked password. Session starts at 2:15 PM.", 24, "login"),
        frame(win("Inbox - ctr.pnguyen@contoso.com - Outlook", [("hrow", ["From", "Subject", "Received"], [26, 54, 20]), ("row", ["Dana Whitfield", "Q3 forecast review", "1:58 PM"], [26, 54, 20], True),
                                                                ("row", ["Jira", "FIN-2231 assigned to you", "1:40 PM"], [26, 54, 20]), ("row", ["Facilities", "Parking lot closure Friday", "11:12 AM"], [26, 54, 20])]),
              "Outlook inbox for ctr.pnguyen@contoso.com lists 'Q3 forecast review' from Dana Whitfield, 'FIN-2231 assigned to you' from Jira and a facilities notice.", 8, "email"),
        frame(win("RE: Q3 forecast review - Message (HTML) - Outlook", [("field", "To", "Dana Whitfield"), ("field", "Subject", "RE: Q3 forecast review"), ("t", "Hi Dana, I will update the accrual lines and send v4 by end of day."), ("btns", ["Send", "Discard"], "left")]),
              "Outlook compose window replying to Dana Whitfield, subject 'RE: Q3 forecast review', text about updating accrual lines and sending v4 by end of day.", 8, "email"),
        frame(win("Q3_Forecast_v4.xlsx - Excel", [("hrow", ["Region", "Q3 Revenue", "Q3 Accrual", "Variance"], [25, 25, 25, 25]), ("row", ["North", "1,240,000", "88,500", "+2.1%"], [25, 25, 25, 25]),
                                                  ("row", ["South", "980,500", "71,200", "-0.8%"], [25, 25, 25, 25]), ("row", ["West", "1,115,300", "79,900", "+1.4%"], [25, 25, 25, 25], True)]),
              "Excel workbook Q3_Forecast_v4.xlsx shows regional revenue, accrual and variance rows (North, South, West).", 10, "document_edit"),
        frame(win("Q3_Forecast_v4.xlsx - Excel", [("t", "fx  =SUM(C2:C4)   ->   239,600", "gray"), ("hrow", ["Region", "Q3 Revenue", "Q3 Accrual", "Variance"], [25, 25, 25, 25]), ("row", ["Total", "3,335,800", "239,600", "+0.9%"], [25, 25, 25, 25], True)]),
              "The user edits the Total row in Q3_Forecast_v4.xlsx with the formula =SUM(C2:C4) giving 239,600 accrual and +0.9% variance.", 8, "document_edit"),
        frame(win("FIN-2231 Reconcile Q3 accrual report - Jira - Microsoft Edge", [("h", "FIN-2231: Reconcile Q3 accrual report"), ("field", "Assignee", "Priya Nguyen (contractor)"), ("field", "Status", "In Progress"), ("t", "Update Q3_Forecast_v4.xlsx and send to Dana Whitfield.")],
                  toolbar="https://jira.contoso.local/browse/FIN-2231"),
              "Jira ticket FIN-2231 'Reconcile Q3 accrual report' assigned to Priya Nguyen (contractor), In Progress, task: update Q3_Forecast_v4.xlsx and send to Dana Whitfield.", 8, "browsing"),
        frame(win("Month-end close checklist - Confluence - Microsoft Edge", [("h", "Month-end close checklist"), ("t", "1. Reconcile accrual report  2. Review variance  3. Send forecast to FP&A"), ("t", "Owner: Finance Operations")],
                  toolbar="https://wiki.contoso.local/display/FIN/Month-end+close"),
              "Confluence page 'Month-end close checklist' lists reconcile accrual report, review variance and send forecast to FP&A.", 6, "browsing"),
        frame(win("InvoiceGen Utility - Free Download - Microsoft Edge", [("h", "InvoiceGen Utility"), ("t", "Create and batch-print invoices in one click. Free for personal use."), ("btns", ["Download for Windows", "Learn more"], "left")],
                  toolbar="https://invoicegen-tools.example/download"),
              "Edge shows 'InvoiceGen Utility - Free Download' at invoicegen-tools.example with a Download for Windows button. The tool is not part of the finance task.", 32, "browsing"),
        frame(win("Downloads - Microsoft Edge", [("t", "Downloads", "gray")], toolbar="https://invoicegen-tools.example/download"),
              "Edge shows invoicegen_setup.exe downloaded (3.4 MB) and a SmartScreen 'Windows protected your PC - unrecognized app' prompt.", 34, "file_transfer",
              dialog=dlg("Windows protected your PC", [("h", "Windows protected your PC"), ("t", "Microsoft Defender SmartScreen prevented an unrecognized app from starting."), ("t", "App: invoicegen_setup.exe   Publisher: Unknown publisher"), ("btns", ["Don't run", "Run anyway"])], (640, 280))),
        frame(win("Windows Security", [("h", "Threats found"), ("t", "Threat blocked: PUA:Win32/InvoiceGen", "red"), ("t", "File: C:\\Users\\ctr.pnguyen\\Downloads\\invoicegen_setup.exe - quarantined"), ("btns", ["Actions", "Close"], "left")], rect=(260, 90, 1020, 520)),
              "Windows Security reports 'Threat blocked: PUA:Win32/InvoiceGen'; C:\\Users\\ctr.pnguyen\\Downloads\\invoicegen_setup.exe was quarantined.", 38, "other"),
        frame(win("Windows Security - Virus & threat protection settings", [("h", "Virus & threat protection settings"), ("toggle", "Real-time protection", True), ("toggle", "Cloud-delivered protection", True), ("toggle", "Automatic sample submission", True)]),
              "Windows Security > Virus & threat protection settings: Real-time protection is On; the user is about to change it. Cloud-delivered protection and Automatic sample submission are On.", 40, "config_change"),
        frame(win("Windows Security - Virus & threat protection settings", [("h", "Virus & threat protection settings"), ("t", "Real-time protection is off, leaving your device vulnerable.", "red"), ("toggle", "Real-time protection", False), ("toggle", "Cloud-delivered protection", True)]),
              "Real-time protection is switched Off in Windows Security with the warning 'Real-time protection is off, leaving your device vulnerable.'", 48, "config_change"),
        frame(listing("Downloads - File Explorer", r"C:\Users\ctr.pnguyen\Downloads", [("invoicegen_setup.exe", "3,482 KB", "10/14/2025 2:31 PM"), ("Q3_Forecast_v3.xlsx", "212 KB", "10/10/2025 4:12 PM")], 0),
              "File Explorer at Downloads shows invoicegen_setup.exe (3,482 KB) restored and selected, next to Q3_Forecast_v3.xlsx.", 36, "other"),
        frame(win("User Account Control", [("t", "Downloads", "gray")]), "A User Account Control prompt asks 'Do you want to allow this app from an unknown publisher to make changes to your device?' for invoicegen_setup.exe, Verified publisher: Unknown.", 44, "config_change",
              dialog=dlg("User Account Control", [("h", "Do you want to allow this app to make changes?"), ("t", "invoicegen_setup.exe"), ("t", "Verified publisher: Unknown"), ("btns", ["Yes", "No"])], (600, 260))),
        frame(win("InvoiceGen Setup", [("h", "InvoiceGen Setup"), ("progress", "Installing InvoiceGen Utility", 100), ("t", "Installation complete. Launch InvoiceGen Utility"), ("btns", ["Finish"])], rect=(300, 110, 980, 520)),
              "InvoiceGen Setup wizard reports installation complete with a Finish button; an unrecognised third-party program was installed on the workstation.", 44, "other"),
        frame(win("Windows Security - Virus & threat protection settings", [("h", "Virus & threat protection settings"), ("t", "Your device is being protected.", "green"), ("toggle", "Real-time protection", True), ("toggle", "Cloud-delivered protection", True)]),
              "Real-time protection is turned back On and the page reads 'Your device is being protected.' It was off for roughly 12 seconds.", 28, "config_change"),
        frame(win("Q3_Forecast_v4.xlsx - Excel", [("hrow", ["Region", "Q3 Revenue", "Q3 Accrual", "Variance"], [25, 25, 25, 25]), ("row", ["Total", "3,335,800", "239,600", "+0.9%"], [25, 25, 25, 25], True)]),
              "Excel returns to Q3_Forecast_v4.xlsx with the Total row (3,335,800 revenue, 239,600 accrual, +0.9%).", 8, "document_edit"),
        frame(listing("This PC > \\\\FILESRV01 - File Explorer", r"\\FILESRV01", [("Finance$", "Share", "Available"), ("HR$", "Share", "Available"), ("IT$", "Share", "Available"), ("Public", "Share", "Available")], 1, hdr=("Name", "Type", "Status")),
              "File Explorer at \\\\FILESRV01 lists network shares Finance$, HR$, IT$ and Public with HR$ selected. The HR share is unrelated to the finance task.", 40, "other"),
        frame(listing("Compensation - File Explorer", r"\\FILESRV01\HR$\Compensation", [("Salary_Bands_2025.xlsx", "86 KB", "09/30/2025"), ("Exec_Comp_2025.xlsx", "112 KB", "09/30/2025"), ("Bonus_Pool_Draft.docx", "64 KB", "10/02/2025")], 0),
              "File Explorer at \\\\FILESRV01\\HR$\\Compensation shows Salary_Bands_2025.xlsx, Exec_Comp_2025.xlsx and Bonus_Pool_Draft.docx; Salary_Bands_2025.xlsx is selected.", 48, "other"),
        frame(win("Salary_Bands_2025.xlsx - Excel", [("hrow", ["Level", "Min", "Mid", "Max"], [25, 25, 25, 25]), ("row", ["L3", "78,000", "92,000", "106,000"], [25, 25, 25, 25]), ("row", ["L4", "98,000", "116,000", "134,000"], [25, 25, 25, 25]),
                                                    ("row", ["L5", "124,000", "146,000", "168,000"], [25, 25, 25, 25])]),
              "Excel opens Salary_Bands_2025.xlsx from the HR share showing salary bands by level (L3-L5 with min, mid and max). Compensation data is visible and not related to FIN-2231.", 52, "document_edit"),
        frame(listing("Q3 - File Explorer", r"\\FILESRV01\Finance$\Q3", [("Q3_Forecast_v4.xlsx", "214 KB", "10/14/2025 3:02 PM"), ("Q3_Accruals.csv", "88 KB", "10/12/2025")], 0),
              "File Explorer returns to \\\\FILESRV01\\Finance$\\Q3 with Q3_Forecast_v4.xlsx (214 KB, 3:02 PM) selected.", 10, "other"),
        frame(win("Q3 forecast v4 - Message (HTML) - Outlook", [("field", "To", "Dana Whitfield"), ("field", "Subject", "Q3 forecast v4"), ("field", "Attached", "Q3_Forecast_v4.xlsx (214 KB)"), ("btns", ["Send", "Discard"], "left")]),
              "Outlook message to Dana Whitfield, subject 'Q3 forecast v4', with attachment Q3_Forecast_v4.xlsx (214 KB). This is the expected deliverable for FIN-2231.", 10, "email"),
        frame(win("Windows", [("t", "WS-FIN-12", "gray")]), "A 'Sign out' prompt is shown for WS-FIN-12 at about 3:27 PM.", 6, "other",
              dialog=dlg("Sign out", [("h", "Sign out of WS-FIN-12?"), ("btns", ["Sign out", "Cancel"])], (460, 200))),
        frame(ended("Your Remote Desktop Services session has ended. Reason: You have been logged off by the user."), "Remote Desktop Connection shows 'Session ended: logged off by the user'.", 3, "idle"),
    ],
}

SESSIONS = [SESSION_A, SESSION_B, SESSION_C]
