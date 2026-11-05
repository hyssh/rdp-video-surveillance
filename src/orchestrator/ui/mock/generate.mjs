// Generates mock/sessions.json (3 demo sessions, 25 frames @ 3s). Run: npm run mock:generate
import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const dir = dirname(fileURLToPath(import.meta.url));

const S1 = [
  ['login', 10, 'RDP session starts; vendor authenticates to jump host JUMP-HOST-01 and completes an MFA prompt.'],
  ['login', 12, 'Windows desktop loads and the vendor opens the Start menu.'],
  ['browsing', 15, 'Vendor opens ServiceNow change ticket CHG-48213 in Edge to confirm the approved maintenance window.'],
  ['terminal', 20, 'PowerShell window opened; vendor runs hostname and whoami.'],
  ['admin_console', 30, 'SQL Server Management Studio launched and connected to PROD-SQL-02 with a sysadmin login.'],
  ['admin_console', 35, 'Object Explorer expanded; vendor browses customer_db tables including dbo.Customers and dbo.PaymentCards.'],
  ['admin_console', 45, 'Query window opened; SELECT TOP 1000 * FROM dbo.Customers executed (not in the change ticket scope).'],
  ['admin_console', 58, 'Query results show SSN and email columns in clear text on screen.'],
  ['config_change', 62, 'Vendor opens Security > Logins and inspects the sa and svc_backup accounts.'],
  ['config_change', 72, "Creates a new SQL login 'tmp_support' and assigns the sysadmin server role."],
  ['config_change', 78, "'Enforce password policy' is unchecked for the new tmp_support login."],
  ['terminal', 70, 'PowerShell: net localgroup administrators tmp_support /add executed on the database host.'],
  ['terminal', 80, 'Windows Defender real-time monitoring disabled via Set-MpPreference -DisableRealtimeMonitoring.'],
  ['file_transfer', 85, 'bcp exports dbo.PaymentCards to C:\\Users\\Public\\export.csv.'],
  ['file_transfer', 92, 'export.csv compressed into a password-protected export.zip using 7-Zip.'],
  ['file_transfer', 95, 'Vendor opens a personal OneDrive in the browser and uploads export.zip to external cloud storage.'],
  ['browsing', 90, 'Upload reaches 100%; an anyone-with-the-link share URL is generated.'],
  ['terminal', 88, 'Security event log cleared with wevtutil cl Security.'],
  ['terminal', 84, 'export.csv deleted and cipher /w run to wipe free space (anti-forensics).'],
  ['admin_console', 55, 'SSMS reopened to check SQL Agent job history for backup jobs.'],
  ['browsing', 40, "Vendor returns to ticket CHG-48213 and marks the work 'Completed - no issues'."],
  ['idle', 15, 'Session idle; desktop with no active windows.'],
  ['idle', 8, 'Session idle; screen unchanged.'],
  ['login', 10, 'Vendor signs out of the Windows session.'],
  ['idle', 5, 'RDP disconnect dialog; session closes.'],
];
const S2 = [
  ['login', 4, 'Vendor connects to patch-mgmt server WSUS-01 with a named service account; MFA approved.'],
  ['login', 5, 'Desktop loads; Server Manager opens automatically.'],
  ['browsing', 8, 'Vendor opens change ticket CHG-48190 (monthly OS patching) in Edge.'],
  ['admin_console', 12, 'Windows Server Update Services console opened; Updates > Critical Updates listed.'],
  ['admin_console', 14, 'KB5034441 and KB5034439 selected for approval to the Production-Servers group.'],
  ['admin_console', 16, 'Approval dialog confirms install deadline for Saturday 02:00, inside the change window.'],
  ['admin_console', 14, 'Update status report shows 42 of 45 servers compliant.'],
  ['terminal', 18, 'PowerShell: Get-HotFix | Sort InstalledOn -Desc | Select -First 5 on WSUS-01.'],
  ['terminal', 20, 'PowerShell: Invoke-Command to restart the Windows Update service on three lagging servers.'],
  ['terminal', 18, 'Command completes without errors; services report Running.'],
  ['admin_console', 12, 'WSUS report refreshed; 45 of 45 servers pending install.'],
  ['browsing', 8, 'Vendor opens the internal runbook wiki page for post-patch validation.'],
  ['browsing', 6, 'Reads checklist items; no external sites accessed.'],
  ['terminal', 14, 'Test-NetConnection to the monitoring server; ping succeeds.'],
  ['admin_console', 10, 'Event Viewer > Windows Update Client log reviewed; no failures.'],
  ['browsing', 8, "Ticket CHG-48190 updated with patch list and marked 'Implemented'."],
  ['idle', 4, 'Session idle; Server Manager dashboard visible.'],
  ['idle', 3, 'Session idle.'],
  ['browsing', 6, 'Vendor attaches the WSUS compliance report PDF to the ticket.'],
  ['file_transfer', 22, 'Compliance report copied to the approved SharePoint change-evidence library.'],
  ['browsing', 8, 'Ticket status confirmed as updated.'],
  ['idle', 3, 'Session idle.'],
  ['login', 5, 'Vendor signs out of the session.'],
  ['idle', 3, 'RDP disconnect dialog.'],
  ['idle', 2, 'Session ended.'],
];
const S3 = [
  ['login', 8, 'Contractor connects to the shared analytics VM using a named contractor account.'],
  ['login', 10, 'Windows desktop loads; Outlook and Teams start automatically.'],
  ['browsing', 14, 'Contractor opens the internal wiki and the BI dashboard documentation.'],
  ['browsing', 18, 'Power BI service opened; Sales Overview report loaded.'],
  ['admin_console', 25, 'Azure Data Studio connected to analytics-sql with a read-only login.'],
  ['admin_console', 30, 'Query: SELECT region, SUM(revenue) FROM sales_summary GROUP BY region.'],
  ['admin_console', 28, 'Results rendered as a bar chart; no sensitive columns present.'],
  ['browsing', 32, 'Contractor opens a personal webmail tab alongside the corporate dashboard.'],
  ['browsing', 36, 'Webmail inbox visible; a draft with an attachment named sales_q3.xlsx is open.'],
  ['file_transfer', 48, 'sales_q3.xlsx downloaded from the BI portal to the Downloads folder.'],
  ['file_transfer', 55, 'File attached to the personal webmail draft (not a corporate mailbox).'],
  ['file_transfer', 62, 'Webmail draft sent with sales_q3.xlsx to an external address.'],
  ['browsing', 45, 'Contractor opens a file-sharing website and browses the upload page.'],
  ['browsing', 38, 'Closes the file-sharing site without uploading.'],
  ['admin_console', 30, 'Returns to Azure Data Studio; runs a second read-only query on sales_summary.'],
  ['terminal', 35, 'Windows Terminal: pip install pandas openpyxl run from an unapproved package index.'],
  ['terminal', 40, 'Python script executed to aggregate query results locally.'],
  ['terminal', 28, 'Script output saved to C:\\Users\\contractor\\Documents\\agg.csv.'],
  ['browsing', 22, 'Contractor reads internal documentation on the data catalog.'],
  ['browsing', 18, 'Teams chat window opened; discussing report requirements with an employee.'],
  ['idle', 12, 'Session idle; no activity.'],
  ['idle', 10, 'Session idle.'],
  ['browsing', 14, 'Contractor reviews the BI dashboard one last time.'],
  ['login', 8, 'Contractor signs out.'],
  ['idle', 4, 'RDP disconnect dialog.'],
];

const TEMPLATES = {
  login: [['title_bar', 'Remote Desktop Connection', [0, 0, 1, 0.05]], ['text_field', 'Username', [0.35, 0.35, 0.65, 0.42]], ['text_field', 'Password', [0.35, 0.45, 0.65, 0.52]], ['button', 'Sign in', [0.42, 0.56, 0.58, 0.63]], ['icon', 'Taskbar', [0, 0.95, 1, 1]]],
  browsing: [['title_bar', 'Microsoft Edge', [0, 0, 1, 0.05]], ['text_field', 'Address bar', [0.1, 0.07, 0.8, 0.12]], ['menu', 'Navigation menu', [0.02, 0.15, 0.2, 0.9]], ['text', 'Page content', [0.24, 0.2, 0.95, 0.85]], ['button', 'Submit', [0.8, 0.8, 0.94, 0.88]]],
  terminal: [['title_bar', 'Windows PowerShell', [0.05, 0.08, 0.95, 0.14]], ['text', 'Console output', [0.05, 0.15, 0.95, 0.85]], ['text_field', 'Command prompt', [0.05, 0.86, 0.95, 0.92]], ['icon', 'Taskbar', [0, 0.95, 1, 1]]],
  admin_console: [['title_bar', 'SQL Server Management Studio', [0, 0, 1, 0.05]], ['menu', 'Object Explorer', [0.01, 0.08, 0.22, 0.9]], ['text_field', 'Query editor', [0.24, 0.1, 0.99, 0.45]], ['text', 'Results grid', [0.24, 0.48, 0.99, 0.9]], ['button', 'Execute', [0.25, 0.06, 0.32, 0.095]]],
  config_change: [['title_bar', 'Login Properties', [0.15, 0.08, 0.85, 0.14]], ['text_field', 'Login name', [0.2, 0.2, 0.8, 0.27]], ['checkbox', 'Enforce password policy', [0.2, 0.32, 0.6, 0.38]], ['checkbox', 'sysadmin role', [0.2, 0.42, 0.6, 0.48]], ['button', 'OK', [0.62, 0.82, 0.74, 0.88]], ['button', 'Cancel', [0.76, 0.82, 0.88, 0.88]]],
  file_transfer: [['title_bar', 'File Explorer', [0, 0, 1, 0.05]], ['text_field', 'Path', [0.1, 0.07, 0.9, 0.12]], ['icon', 'File: export', [0.28, 0.25, 0.4, 0.42]], ['button', 'Upload', [0.8, 0.8, 0.94, 0.88]], ['text', 'Transfer progress', [0.28, 0.6, 0.9, 0.7]]],
  idle: [['icon', 'Recycle Bin', [0.02, 0.03, 0.08, 0.13]], ['icon', 'Taskbar', [0, 0.95, 1, 1]], ['icon', 'Clock', [0.92, 0.95, 1, 1]]],
  other: [['icon', 'Taskbar', [0, 0.95, 1, 1]]],
};

const level = (s) => (s >= 75 ? 'critical' : s >= 50 ? 'high' : s >= 30 ? 'medium' : 'low');

function build(vid, filename, rows, summary, startedAt) {
  const frames = rows.map(([action, risk, desc], i) => ({
    id: vid * 100 + i + 1,
    t: i * 3,
    action_type: action,
    risk_score: risk,
    risk_level: level(risk),
    description: desc,
    width: 1280,
    height: 720,
    elements: (TEMPLATES[action] || TEMPLATES.other).map(([type, content, bbox], j) => ({
      type,
      content: j === 0 && action !== 'idle' ? content : content,
      interactivity: ['button', 'text_field', 'checkbox', 'menu'].includes(type),
      bbox,
    })),
  }));
  const overall = Math.round(Math.max(...rows.map((r) => r[1])) * 0.6 + (rows.reduce((a, r) => a + r[1], 0) / rows.length) * 0.4);
  return {
    id: vid, filename, duration_s: rows.length * 3, interval_s: 3, summary, risk_score: overall,
    security_level: level(overall), created_at: startedAt, has_clip: false, frames,
  };
}

const sessions = [
  build(1, 'vendor_db_admin_session.mp4', S1,
    '**Risk: CRITICAL.** A vendor admin, working under change ticket CHG-48213, went far beyond the approved scope: queried customer PII, created a hidden sysadmin login (tmp_support), disabled Windows Defender, exported the PaymentCards table, uploaded it to personal cloud storage and then cleared the Security event log. Recommend immediate account suspension, credential rotation and incident response.',
    '2025-01-14T14:02:11Z'),
  build(2, 'vendor_routine_patching.mp4', S2,
    '**Risk: LOW.** Routine monthly patching on WSUS-01 matched change ticket CHG-48190. Only approved Microsoft updates were approved and validated; evidence was filed to the sanctioned SharePoint library. No anomalies observed.',
    '2025-01-15T09:30:02Z'),
  build(3, 'contractor_mixed_session.mp4', S3,
    '**Risk: MEDIUM.** A contractor mostly performed legitimate read-only BI analysis, but also downloaded a sales workbook and sent it via personal webmail, browsed a file-sharing site and installed packages from an unapproved index. Needs a manager follow-up on the external email.',
    '2025-01-16T16:45:40Z'),
];
writeFileSync(join(dir, 'sessions.json'), JSON.stringify(sessions, null, 1));
console.log('wrote sessions.json', sessions.map((s) => `${s.filename}:${s.risk_score}`).join(' '));
