export default async (request) => {
  if (request.method !== 'POST') {
    return json({ message: 'Method not allowed' }, 405);
  }

  const apiKey = process.env.BREVO_API_KEY;
  const senderEmail = process.env.BREVO_SENDER_EMAIL;
  const senderName = process.env.BREVO_SENDER_NAME || 'TALENT.PREMIUM';

  if (!apiKey || !senderEmail) {
    return json({ message: 'Email service is not configured.' }, 500);
  }

  let payload;
  try {
    payload = await request.json();
  } catch {
    return json({ message: 'Invalid JSON body.' }, 400);
  }

  const { toEmail, toName, subject, bodyContent } = payload || {};
  if (!toEmail || !subject || !bodyContent) {
    return json({ message: 'Missing required email fields.' }, 400);
  }

  const response = await fetch('https://api.brevo.com/v3/smtp/email', {
    method: 'POST',
    headers: {
      accept: 'application/json',
      'api-key': apiKey,
      'content-type': 'application/json',
    },
    body: JSON.stringify({
      sender: { name: senderName, email: senderEmail },
      to: [{ email: toEmail, name: toName || toEmail }],
      subject,
      htmlContent: bodyContent,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    return json(
      { message: error.message || response.statusText || 'Email delivery failed.' },
      response.status
    );
  }

  return json({ ok: true });
};

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}
